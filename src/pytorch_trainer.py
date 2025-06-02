"""
Trainer module for handling PyTorch model training and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import os
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt

from src.pytorch_model import (
    ModelFactory, 
    TimeSeriesDataModule, 
    train_model, 
    evaluate_model
)
from src.data_preprocessing import MultiPointPreprocessor
from src.logger import TrainingLogger
from src.visualization import plot_sequence_prediction_comparison, plot_predictions
from src.clustering import TimeSeriesClusterer


class MultiPointPyTorchTrainer:
    """Handles training and evaluation of multi-point time series models using PyTorch."""
    
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        param_space: Dict[str, List[Any]],
        sequence_length: int,
        output_sequence_length: int,
        time_features: List[str] = None,
        normalization_range: Tuple[int, int] = (-1, 1),
        output_dir: str = "./outputs",
        log_dir: str = "./logs",
        n_iter: int = 5,
        model_type: str = 'lstm',
        clusterer: Optional[TimeSeriesClusterer] = None,
        min_correlation: float = 0.7,
        num_workers: int = 4,
        gpu_enabled: bool = True
    ):
        """
        Initialize trainer with data and parameters.
        
        Args:
            train_data: Training data DataFrame
            test_data: Test data DataFrame
            param_space: Dictionary of parameter spaces for optimization
            sequence_length: Length of input sequences
            output_sequence_length: Length of output sequences
            time_features: List of time-based features to include
            normalization_range: Range for data normalization
            output_dir: Directory to save outputs
            log_dir: Directory to save logs
            n_iter: Number of iterations for Bayesian optimization
            model_type: Type of model to use ('lstm')
            clusterer: Optional clusterer instance to use for point selection
            min_correlation: Minimum correlation threshold for point selection
            num_workers: Number of workers for data loading
            gpu_enabled: Whether to use GPU if available
        """
        self.train_data = train_data
        self.test_data = test_data
        self.param_space = param_space
        self.sequence_length = sequence_length
        self.output_sequence_length = output_sequence_length
        self.time_features = time_features
        self.normalization_range = normalization_range
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.n_iter = n_iter
        self.model_type = model_type
        self.clusterer = clusterer
        self.min_correlation = min_correlation
        self.num_workers = num_workers
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.gpus = 1 if self.gpu_enabled else 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize storage for results
        self.model = None
        self.best_params = None
        self.best_metrics = None
        self.history_dict = {}
        self.predictions = None
        self.all_predictions = None
        
        # Initialize preprocessor
        self.preprocessor = MultiPointPreprocessor(
            sequence_length=sequence_length,
            output_sequence_length=output_sequence_length,
            normalization_range=normalization_range,
            time_features=time_features
        )
        
        # Initialize logger
        self.logger = TrainingLogger(
            name="multi_point_pytorch_trainer",
            log_dir=log_dir
        )
        
    def _create_model_and_train(self, **params) -> float:
        """
        Create and train model with given parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Validation RMSE score
        """
        try:
            # Convert batch_size to regular Python integer
            params['batch_size'] = int(params['batch_size'])
            
            # Prepare model configuration based on type
            if self.model_type == 'lstm':
                # Convert string parameters to tuples
                lstm_units_str = params['lstm_units']
                dropout_rates_str = params['dropout_rates']
                dense_units_str = params['dense_units']
                
                # Convert strings to tuples of appropriate types
                lstm_units_tuple = tuple(int(x) for x in lstm_units_str.split('_'))
                dropout_rates_list = [float(x) for x in dropout_rates_str.split('_')]
                
                # Ensure dropout_rates matches lstm_units length
                if len(dropout_rates_list) < len(lstm_units_tuple):
                    # If we have fewer dropout rates, repeat the last one
                    last_dropout = dropout_rates_list[-1]
                    while len(dropout_rates_list) < len(lstm_units_tuple):
                        dropout_rates_list.append(last_dropout)
                elif len(dropout_rates_list) > len(lstm_units_tuple):
                    # If we have more dropout rates, truncate
                    dropout_rates_list = dropout_rates_list[:len(lstm_units_tuple)]
                
                dropout_rates_tuple = tuple(dropout_rates_list)
                dense_units_tuple = tuple(int(x) for x in dense_units_str.split('_'))
                
                config = {
                    'num_points': self.num_points,
                    'sequence_length': self.sequence_length,
                    'num_features': self.num_features,
                    'output_sequence_length': self.output_sequence_length,
                    'lstm_units': lstm_units_tuple,
                    'dropout_rates': dropout_rates_tuple,
                    'dense_units': dense_units_tuple,
                    'learning_rate': params['learning_rate'],
                    'point_embedding_dim': params['point_embedding_dim'],
                    'use_point_embeddings': params['use_point_embeddings'],
                    'use_residual_connections': params['use_residual_connections'],
                    'weight_decay': params.get('weight_decay', 1e-5),
                    'lr_scheduler_factor': params.get('lr_scheduler_factor', 0.5),
                    'lr_scheduler_patience': params.get('lr_scheduler_patience', 10)
                }
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Create model using factory
            model = ModelFactory.create_model(self.model_type, config)
            
            # Ensure data is properly shaped and contiguous
            x_train = torch.from_numpy(self.x_train).float()
            x_val = torch.from_numpy(self.x_val).float()
            y_train = torch.from_numpy(self.y_train).float()
            y_val = torch.from_numpy(self.y_val).float()
            indices_train = torch.from_numpy(self.indices_train).long()
            indices_val = torch.from_numpy(self.indices_val).long()
            
            # Create data module
            data_module = TimeSeriesDataModule(
                x_train=x_train,
                y_train=y_train,
                point_indices_train=indices_train,
                batch_size=params['batch_size'],
                x_val=x_val,
                y_val=y_val,
                point_indices_val=indices_val,
                num_workers=self.num_workers,
                pin_memory=self.gpu_enabled
            )
            
            # Train model using the training utility
            trained_model = train_model(
                model=model,
                data_module=data_module,
                max_epochs=int(params['epochs']),
                early_stopping_patience=10,
                checkpoint_dir=os.path.join(self.output_dir, 'checkpoints'),
                model_name=f"run_{self.n_iter}",
                gpus=self.gpus
            )
            
            # Evaluate on validation set
            val_metrics = evaluate_model(
                model=trained_model,
                data_module=data_module,
                gpus=self.gpus
            )
            
            # Get validation loss (RMSE)
            val_rmse = val_metrics.get('test_mse', None)
            
            # If no validation RMSE, calculate it manually
            if val_rmse is None:
                # Create validation DataLoader
                val_dataloader = data_module.val_dataloader()
                
                # Get predictions
                trainer = pl.Trainer(
                    accelerator='gpu' if self.gpus > 0 else 'cpu',
                    devices=self.gpus if self.gpus > 0 else None
                )
                predictions = trainer.predict(trained_model, dataloaders=val_dataloader)
                
                # Concatenate predictions
                val_predictions = torch.cat(predictions, dim=0).cpu().numpy()
                
                # Calculate RMSE
                val_rmse = np.sqrt(np.mean((self.y_val - val_predictions) ** 2))
            
            # Check for invalid values in RMSE
            if np.isnan(val_rmse) or np.isinf(val_rmse) or val_rmse > 1e6:
                self.logger.warning(f"Invalid RMSE ({val_rmse}) with parameters: {params}")
                return 1e6
            
            # Save model and parameters if this is the best RMSE
            if self.model is None or val_rmse < self.best_val_rmse:
                self.model = trained_model
                self.best_params = params
                self.best_val_rmse = val_rmse
                
                # Save model
                model_save_path = os.path.join(self.output_dir, 'best_model')
                os.makedirs(model_save_path, exist_ok=True)
                trained_model.save(model_save_path, "best_model")
                
                # Log best parameters
                self.logger.info(f"New best parameters: {params}, RMSE: {val_rmse}")
            
            # Store current run in history
            self.history_dict[str(params)] = {
                'rmse': val_rmse,
                'params': params
            }
            
            return val_rmse
            
        except Exception as e:
            self.logger.error(f"Error in _create_model_and_train: {str(e)}", exc_info=True)
            return 1e6
    
    def _optimize_hyperparameters(self):
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Returns:
            Tuple of (best_params, best_score)
        """
        self.logger.info("Starting hyperparameter optimization")
        
        # Create search space for Bayesian optimization based on model type
        search_space = []
        
        if self.model_type == 'lstm':
            # Helper function to handle different parameter formats
            def get_range(param, default_low, default_high):
                if isinstance(param, (list, tuple)):
                    if len(param) >= 2:
                        return min(param), max(param)
                    elif len(param) == 1:
                        # If single value, create a range around it
                        value = param[0]
                        # For batch_size, ensure it doesn't exceed reasonable limits
                        if isinstance(value, (int, np.integer)):
                            # Calculate range around the value, but ensure it's within reasonable bounds
                            low = max(default_low, value // 2)
                            high = min(default_high, value * 2)
                            # If high is less than low, adjust the range
                            if high < low:
                                high = max(default_high, value)
                                low = min(default_low, value)
                            return low, high
                        else:
                            return default_low, default_high
                    else:
                        return default_low, default_high
                else:
                    # If single value, create a range around it
                    value = param
                    # For batch_size, ensure it doesn't exceed reasonable limits
                    if isinstance(value, (int, np.integer)):
                        # Calculate range around the value, but ensure it's within reasonable bounds
                        low = max(default_low, value // 2)
                        high = min(default_high, value * 2)
                        # If high is less than low, adjust the range
                        if high < low:
                            high = max(default_high, value)
                            low = min(default_low, value)
                        return low, high
                    else:
                        return default_low, default_high
            
            # Use the helper function to get ranges for numerical parameters
            # Limit batch_size to reasonable values (max 512)
            batch_size_range = get_range(self.param_space.get('batch_size', [32, 128]), 32, 512)
            
            # Log the ranges for debugging
            self.logger.info(f"Batch size range: {batch_size_range}")
            
            # Ensure batch_size is within reasonable limits
            if batch_size_range[1] > 512:
                self.logger.warning(f"Batch size upper bound ({batch_size_range[1]}) exceeds maximum allowed value (512). Adjusting to 512.")
                batch_size_range = (batch_size_range[0], 512)
            
            epochs_range = get_range(self.param_space.get('epochs', [50, 200]), 50, 200)
            point_embedding_dim_range = get_range(self.param_space.get('point_embedding_dim', [4, 16]), 4, 16)
            
            # Ensure learning_rate is in the correct format
            if 'learning_rate' in self.param_space:
                lr_values = self.param_space['learning_rate']
                if isinstance(lr_values, (list, tuple)) and len(lr_values) >= 2:
                    lr_low, lr_high = min(lr_values), max(lr_values)
                else:
                    # Default or single value
                    lr_value = lr_values[0] if isinstance(lr_values, (list, tuple)) else lr_values
                    lr_low, lr_high = lr_value * 0.5, lr_value * 2.0
            else:
                # Default learning rate range
                lr_low, lr_high = 0.0001, 0.01
            
            # Get categorical parameters
            lstm_units = self.param_space.get('lstm_units', ['64_32'])
            if not isinstance(lstm_units, (list, tuple)):
                lstm_units = [str(lstm_units)]
                
            dropout_rates = self.param_space.get('dropout_rates', ['0.2_0.2'])
            if not isinstance(dropout_rates, (list, tuple)):
                dropout_rates = [str(dropout_rates)]
                
            dense_units = self.param_space.get('dense_units', ['32'])
            if not isinstance(dense_units, (list, tuple)):
                dense_units = [str(dense_units)]
                
            use_point_embeddings = self.param_space.get('use_point_embeddings', [True])
            if not isinstance(use_point_embeddings, (list, tuple)):
                use_point_embeddings = [use_point_embeddings]
                
            use_residual_connections = self.param_space.get('use_residual_connections', [True])
            if not isinstance(use_residual_connections, (list, tuple)):
                use_residual_connections = [use_residual_connections]
                
            # Define the search space
            search_space = [
                Categorical(lstm_units, name='lstm_units'),
                Categorical(dropout_rates, name='dropout_rates'),
                Categorical(dense_units, name='dense_units'),
                Real(lr_low, lr_high, prior='log-uniform', name='learning_rate'),
                Integer(batch_size_range[0], batch_size_range[1], name='batch_size'),
                Integer(epochs_range[0], epochs_range[1], name='epochs'),
                Integer(point_embedding_dim_range[0], point_embedding_dim_range[1], name='point_embedding_dim'),
                Categorical(use_point_embeddings, name='use_point_embeddings'),
                Categorical(use_residual_connections, name='use_residual_connections')
            ]
        else:
            raise ValueError(f"Unsupported model type for hyperparameter optimization: {self.model_type}")
                
        # Initialize optimizer
        optimizer = Optimizer(
            dimensions=search_space,
            random_state=42,
            n_initial_points=5  # Increase initial points for better exploration
        )
        
        # Set best RMSE to infinity
        self.best_val_rmse = float('inf')
        
        # Log the search space
        self.logger.info("Search space configuration:")
        for dim in search_space:
            self.logger.info(f"- {dim.name}: {dim}")
        
        with tqdm(total=self.n_iter, desc="Optimizing hyperparameters") as pbar:
            for i in range(self.n_iter):
                x = optimizer.ask()
                
                # Convert to dictionary based on model type
                if self.model_type == 'lstm':
                    params = {
                        'lstm_units': x[0],
                        'dropout_rates': x[1],
                        'dense_units': x[2],
                        'learning_rate': x[3],
                        'batch_size': x[4],
                        'epochs': x[5],
                        'point_embedding_dim': x[6],
                        'use_point_embeddings': x[7],
                        'use_residual_connections': x[8]
                    }
                    
                    # Log current parameters
                    self.logger.info(f"Trying parameters: {params}")
                
                # Evaluate parameters
                score = self._create_model_and_train(**params)
                
                # Update optimizer
                optimizer.tell(x, score)
                
                pbar.update(1)
                pbar.set_postfix(best_score=self.best_val_rmse)
                
                # Log current best parameters
                if self.best_params is not None:
                    self.logger.info(f"Current best parameters: {self.best_params}, RMSE: {self.best_val_rmse}")
        
        # Get best parameters
        best_params = self.best_params
        best_score = self.best_val_rmse
        
        if best_params is None:
            self.logger.warning("Optimization failed to find any valid parameters")
        else:
            self.logger.info(f"Optimization finished. Best parameters: {best_params}, Best RMSE: {best_score}")
        
        return best_params, best_score
    
    def train_and_evaluate(self) -> Tuple[Dict, Dict, pd.DataFrame, Any]:
        """
        Train and evaluate model.
        
        Returns:
            Tuple of (best_params, best_metrics, predictions, model)
        """
        try:
            self.logger.info("Starting training and evaluation")
            
            # Get preprocessed data
            full_train_data = self.train_data.copy()
            full_test_data = self.test_data.copy()
            print(full_train_data)
            print(full_test_data)
            # Perform clustering if requested
            if self.clusterer is not None:
                self.logger.info("Using clusterer for point selection")
                
                # First, select representative points using the entire dataset
                self.logger.info("Selecting representative points from dataset...")
                all_data = pd.concat([self.train_data, self.test_data])
                
                # Store all original columns for later use with all points
                self.all_point_columns = [col for col in all_data.columns if col != 'Date']
                
                # Select representative points using the provided clusterer
                selected_points = self.preprocessor.select_representative_points(
                    all_data,
                    clusterer=self.clusterer,
                    min_correlation=self.min_correlation
                )
                self.logger.info(f"Selected {len(selected_points)} representative points using {self.clusterer.__class__.__name__}")
                
                # Update preprocessor to use only selected points
                self.preprocessor.point_columns = selected_points
            else:
                # Get all non-date columns as point columns
                self.all_point_columns = [col for col in full_train_data.columns if col != 'Date']
                self.preprocessor.point_columns = self.all_point_columns
                self.logger.info(f"Using all {len(self.all_point_columns)} points (no clustering)")
            
            # Transform data
            train_sequences, train_targets, train_indices = self.preprocessor.transform(
                full_train_data
            )
            
            test_sequences, test_targets, test_indices = self.preprocessor.transform(
                full_test_data
            )

            # Log data shapes for debugging
            self.logger.info(f"Train sequences shape: {train_sequences.shape}")
            self.logger.info(f"Train targets shape: {train_targets.shape}")
            self.logger.info(f"Train indices shape: {train_indices.shape}")
            self.logger.info(f"Test sequences shape: {test_sequences.shape}")
            self.logger.info(f"Test targets shape: {test_targets.shape}")
            self.logger.info(f"Test indices shape: {test_indices.shape}")

            # Use test data as validation set
            (self.x_train, self.x_val, self.y_train, self.y_val,
             self.indices_train, self.indices_val) = train_sequences, test_sequences, train_targets, test_targets, train_indices, test_indices

            self.num_points = len(self.preprocessor.point_columns)
            self.num_features = 1 
            print(self.x_train.shape)
            print(self.x_val.shape)
            print(self.y_train.shape)
            print(self.y_val.shape)
            print(self.indices_train.shape)
            print(self.indices_val.shape)
            # ss
            # # Reshape sequences for model input if needed
            # if len(self.x_train.shape) == 2:
            #     # Ensure data is contiguous before reshaping
            #     self.x_train = np.ascontiguousarray(self.x_train)
            #     self.x_val = np.ascontiguousarray(self.x_val)
                
            #     # Reshape to (samples, sequence_length, features)
            #     self.x_train = self.x_train.reshape(-1, self.sequence_length, 1)
            #     self.x_val = self.x_val.reshape(-1, self.sequence_length, 1)
                
            #     # Log shapes after reshaping
            #     self.logger.info(f"Reshaped train sequences shape: {self.x_train.shape}")
            #     self.logger.info(f"Reshaped validation sequences shape: {self.x_val.shape}")
            
            # Calculate maximum valid batch size
            max_batch_size = min(512, len(self.x_train))
            self.logger.info(f"Maximum valid batch size: {max_batch_size}")
            
            # Adjust batch_size in param_space if needed
            if 'batch_size' in self.param_space:
                if isinstance(self.param_space['batch_size'], (list, tuple)):
                    self.param_space['batch_size'] = [min(bs, max_batch_size) for bs in self.param_space['batch_size']]
                else:
                    self.param_space['batch_size'] = min(self.param_space['batch_size'], max_batch_size)
            
            # Run Bayesian optimization
            best_params, best_score = self._optimize_hyperparameters()
            
            # If optimization failed to find good parameters, use defaults
            if best_params is None:
                self.logger.warning("Optimization failed to find good parameters, using defaults")
                best_params = {
                    'batch_size': 32,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'lstm_units': '64_32',
                    'dropout_rates': '0.2_0.2',
                    'dense_units': '32',
                    'point_embedding_dim': 8,
                    'use_point_embeddings': True,
                    'use_residual_connections': True
                }
            
            # Create data module for evaluation with both validation and test data
            data_module = TimeSeriesDataModule(
                x_train=self.x_train,
                y_train=self.y_train,
                point_indices_train=self.indices_train,
                batch_size=best_params['batch_size'],
                x_val=self.x_val,
                y_val=self.y_val,
                point_indices_val=self.indices_val,
                x_test=self.x_val,  # Use validation data as test data for evaluation
                y_test=self.y_val,
                point_indices_test=self.indices_val,
                num_workers=self.num_workers,
                pin_memory=self.gpu_enabled
            )
            
            # Set up the data module for prediction
            data_module.setup()
            
            # Get predictions on validation set
            trainer = pl.Trainer(
                accelerator='gpu' if self.gpus > 0 else 'cpu',
                devices=self.gpus if self.gpus > 0 else None
            )
            predictions = trainer.predict(self.model, dataloaders=data_module.predict_dataloader())
            
            # Concatenate predictions
            val_predictions = torch.cat(predictions, dim=0).cpu().numpy()
            
            # Calculate metrics
            val_rmse = np.sqrt(np.mean((self.y_val - val_predictions) ** 2))
            val_mae = np.mean(np.abs(self.y_val - val_predictions))
            
            # Store metrics
            best_metrics = {
                'val_rmse': val_rmse,
                'val_mae': val_mae
            }
            
            # Create DataFrame for predictions
            # Ensure we only use the actual values that correspond to the prediction sequence length
            actual_values = self.y_val[:, :val_predictions.shape[1]]  # Take only the first N values where N is the prediction sequence length
            print(actual_values.shape)
            print(val_predictions.shape)
            # Create DataFrame with predictions
            predictions_df = pd.DataFrame({
                'actual': [row for row in actual_values],  # Each row contains 5 values
                'predicted': [row for row in val_predictions]  # Each row contains 5 values
            })
            
            self.predictions = predictions_df
            
            return self.best_params, best_metrics, predictions_df, self.model
            
        except Exception as e:
            self.logger.error("Error in train_and_evaluate", exc_info=True)
            raise
    
    def generate_predictions_for_all_test_points(self, generate_plots: bool = True) -> pd.DataFrame:
        """
        Generate predictions for all test points.
        
        Args:
            generate_plots: Whether to generate plots
            
        Returns:
            DataFrame with predictions
        """
        try:
            self.logger.info("Generating predictions for all test points")
            
            if self.model is None:
                raise ValueError("Model must be trained before generating predictions")
            
            # Get all point columns from test data
            point_columns = self.test_data.columns.tolist()
            num_points = len(point_columns)
            
            # Initialize lists to store predictions for each point
            all_predictions = []
            all_actuals = []
            all_point_names = []
            
            # Process each point
            for point_idx, point_col in enumerate(point_columns):
                self.logger.info(f"Processing point {point_idx + 1}/{num_points}: {point_col}")
                
                # Get data for this point
                point_data = self.test_data[point_col].values
                
                # Create sequences for this point
                sequences = []
                for i in range(0, len(point_data) - self.sequence_length):
                    sequence = point_data[i:i + self.sequence_length]
                    if len(sequence) == self.sequence_length:
                        sequences.append(sequence)
                
                if not sequences:
                    self.logger.warning(f"No valid sequences found for point {point_col}")
                    continue
                
                # Convert to numpy array and reshape
                sequences = np.array(sequences)
                if len(sequences.shape) == 2:
                    sequences = sequences.reshape(-1, self.sequence_length, 1)
                
                # Create point indices array
                point_indices = np.full(len(sequences), point_idx).reshape(-1, 1)
                
                # Create data module for prediction
                data_module = TimeSeriesDataModule(
                    x_train=sequences,  # Not used for prediction
                    y_train=np.zeros((len(sequences), self.output_sequence_length)),  # Not used for prediction
                    point_indices_train=point_indices,
                    batch_size=self.best_params['batch_size'],
                    num_workers=self.num_workers,
                    pin_memory=self.gpu_enabled
                )
                
                # Get predictions
                trainer = pl.Trainer(
                    accelerator='gpu' if self.gpus > 0 else 'cpu',
                    devices=self.gpus if self.gpus > 0 else None
                )
                predictions = trainer.predict(self.model, dataloaders=data_module.predict_dataloader())
                
                # Concatenate predictions
                point_predictions = torch.cat(predictions, dim=0).cpu().numpy()
                
                # Get actual values for comparison
                actual_values = point_data[self.sequence_length:self.sequence_length + len(point_predictions)]
                
                # Store predictions and actuals
                all_predictions.append(point_predictions)
                all_actuals.append(actual_values)
                all_point_names.extend([point_col] * len(point_predictions))
            
            # Combine all predictions and actuals
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_actuals = np.concatenate(all_actuals, axis=0)
            
            # Create DataFrame with predictions
            predictions_df = pd.DataFrame({
                'point': all_point_names,
                'actual': [row for row in all_actuals],
                'predicted': [row for row in all_predictions]
            })
            
            self.all_predictions = predictions_df
            
            # Generate plots if requested
            if generate_plots:
                self.logger.info("Generating plots")
                
                # Create output directory for plots
                plots_dir = os.path.join(self.output_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                
                # Plot predictions
                plot_predictions(
                    actual_data=test_targets,
                    predicted_data=test_predictions,
                    output_dir=os.path.join(plots_dir, 'predictions.png')
                )
                
                # Plot sequence comparison
                plot_sequence_prediction_comparison(
                    actual_data=test_targets,
                    predicted_data=test_predictions,
                    n_samples=5,
                    output_dir=os.path.join(plots_dir, 'sequence_comparison.png')
                )

                self._generate_prediction_plots(predictions_df)
            
            return predictions_df
            
        except Exception as e:
            self.logger.error("Error in generate_predictions_for_all_test_points", exc_info=True)
            raise 