"""
Fine-tuning module for model adaptation to specific points.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import os
import logging
from pathlib import Path
from tqdm import tqdm
import gc


class ModelFineTuner:
    """Handles fine-tuning of global models for specific points."""
    
    def __init__(
        self,
        global_model: tf.keras.Model,
        preprocessor: 'MultiPointPreprocessor',
        output_dir: str,
        learning_rate: float = 0.0001,
        fine_tuning_epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 5,
        min_delta: float = 0.001,
        memory_limit: float = 0.9  # Maximum fraction of GPU memory to use
    ):
        """
        Initialize fine-tuner.
        
        Args:
            global_model: Pre-trained global model
            preprocessor: Data preprocessor
            output_dir: Directory to save fine-tuned models
            learning_rate: Learning rate for fine-tuning
            fine_tuning_epochs: Number of fine-tuning epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum change in monitored quantity
            memory_limit: Maximum fraction of GPU memory to use
        """
        self.global_model = global_model
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.fine_tuning_epochs = fine_tuning_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.memory_limit = memory_limit
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU memory management
        self._setup_gpu_memory()
    
    def _setup_gpu_memory(self):
        """Set up GPU memory management."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    # Limit memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set memory limit
                    memory_limit_bytes = int(
                        tf.config.experimental.get_device_details(gpu)['memory_limit'] *
                        self.memory_limit
                    )
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit_bytes
                        )]
                    )
                self.logger.info(f"GPU memory management configured with {self.memory_limit*100}% limit")
            except RuntimeError as e:
                self.logger.error(f"Error setting up GPU memory: {str(e)}")
    
    def _create_point_specific_model(
        self,
        point_index: int,
        strategy: str = 'full'
    ) -> tf.keras.Model:
        """
        Create point-specific model from global model.
        
        Args:
            point_index: Index of the point
            strategy: Fine-tuning strategy ('full', 'last_layer', 'gradual')
            
        Returns:
            Point-specific model
        """
        # Create new model with same architecture
        point_model = tf.keras.models.clone_model(self.global_model)
        point_model.set_weights(self.global_model.get_weights())
        
        # Configure layers for fine-tuning based on strategy
        if strategy == 'last_layer':
            # Freeze all layers except the last few
            for layer in point_model.layers[:-2]:
                layer.trainable = False
                
        elif strategy == 'gradual':
            # Gradually unfreeze layers during training
            for layer in point_model.layers:
                layer.trainable = False
        
        # Compile model with lower learning rate
        point_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return point_model
    
    def _create_callbacks(
        self,
        point_name: str,
        strategy: str
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Create callbacks for fine-tuning.
        
        Args:
            point_name: Name of the point
            strategy: Fine-tuning strategy
            
        Returns:
            List of callbacks
        """
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=self.min_delta,
                restore_best_weights=True
            ),
            
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.output_dir / f"{point_name}_{strategy}_best.h5",
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.early_stopping_patience // 2,
                min_delta=self.min_delta
            ),
            
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=self.output_dir / 'logs' / point_name,
                histogram_freq=1
            )
        ]
        
        # Add callback for gradual unfreezing if using gradual strategy
        if strategy == 'gradual':
            callbacks.append(
                self.GradualUnfreezing(
                    total_epochs=self.fine_tuning_epochs,
                    logger=self.logger
                )
            )
        
        return callbacks
    
    class GradualUnfreezing(tf.keras.callbacks.Callback):
        """Callback for gradual unfreezing of layers during training."""
        
        def __init__(self, total_epochs: int, logger: logging.Logger):
            """Initialize callback."""
            super().__init__()
            self.total_epochs = total_epochs
            self.logger = logger
            self.unfreeze_epoch = total_epochs // 3  # Start unfreezing after 1/3 of training
        
        def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
            """Unfreeze layers gradually."""
            if epoch >= self.unfreeze_epoch:
                trainable_layers = len([l for l in self.model.layers if hasattr(l, 'trainable')])
                layers_per_epoch = max(1, trainable_layers // (self.total_epochs - self.unfreeze_epoch))
                
                start_idx = (epoch - self.unfreeze_epoch) * layers_per_epoch
                end_idx = start_idx + layers_per_epoch
                
                for layer in self.model.layers[start_idx:end_idx]:
                    if hasattr(layer, 'trainable'):
                        layer.trainable = True
                        self.logger.info(f"Unfroze layer: {layer.name}")
    
    def fine_tune_point(
        self,
        point_name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        point_index: int,
        strategy: str = 'full'
    ) -> Tuple[tf.keras.Model, Dict[str, float]]:
        """
        Fine-tune model for specific point.
        
        Args:
            point_name: Name of the point
            train_data: Training data
            test_data: Test data
            point_index: Index of the point
            strategy: Fine-tuning strategy ('full', 'last_layer', 'gradual')
            
        Returns:
            Tuple of (fine-tuned model, evaluation metrics)
        """
        self.logger.info(f"Fine-tuning model for point: {point_name}")
        
        try:
            # Create point-specific model
            point_model = self._create_point_specific_model(point_index, strategy)
            
            # Prepare data
            x_train, y_train = self.preprocessor.prepare_point_data(
                train_data,
                point_name,
                point_index
            )
            
            x_test, y_test = self.preprocessor.prepare_point_data(
                test_data,
                point_name,
                point_index
            )
            
            # Create callbacks
            callbacks = self._create_callbacks(point_name, strategy)
            
            # Fine-tune model
            history = point_model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.fine_tuning_epochs,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            evaluation = point_model.evaluate(x_test, y_test, verbose=0)
            metrics = dict(zip(point_model.metrics_names, evaluation))
            
            # Save training history
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(self.output_dir / f"{point_name}_{strategy}_history.csv")
            
            # Clean up memory
            tf.keras.backend.clear_session()
            gc.collect()
            
            return point_model, metrics
            
        except Exception as e:
            self.logger.error(f"Error fine-tuning model for {point_name}: {str(e)}")
            raise
    
    def fine_tune_all_points(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        point_names: List[str],
        strategies: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Union[tf.keras.Model, Dict[str, float]]]]:
        """
        Fine-tune model for all points.
        
        Args:
            train_data: Training data
            test_data: Test data
            point_names: List of point names
            strategies: List of strategies to try for each point
            
        Returns:
            Dictionary of results for each point
        """
        if strategies is None:
            strategies = ['full', 'last_layer', 'gradual']
        
        results = {}
        
        for point_idx, point_name in enumerate(tqdm(point_names, desc="Fine-tuning points")):
            point_results = {}
            
            for strategy in strategies:
                self.logger.info(f"Trying strategy '{strategy}' for point {point_name}")
                
                try:
                    model, metrics = self.fine_tune_point(
                        point_name,
                        train_data,
                        test_data,
                        point_idx,
                        strategy
                    )
                    
                    point_results[strategy] = {
                        'model': model,
                        'metrics': metrics
                    }
                    
                except Exception as e:
                    self.logger.error(
                        f"Error with strategy '{strategy}' for point {point_name}: {str(e)}"
                    )
                    continue
            
            # Select best strategy based on validation loss
            best_strategy = min(
                point_results.keys(),
                key=lambda s: point_results[s]['metrics']['val_loss']
            )
            
            results[point_name] = point_results[best_strategy]
            self.logger.info(f"Best strategy for {point_name}: {best_strategy}")
            
            # Save results summary
            summary = pd.DataFrame([{
                'point': point_name,
                'best_strategy': best_strategy,
                **point_results[best_strategy]['metrics']
            }])
            summary.to_csv(self.output_dir / f"{point_name}_summary.csv", index=False)
        
        return results 