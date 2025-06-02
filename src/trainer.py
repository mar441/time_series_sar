"""
Trainer module for handling model training and evaluation.
"""

import os
# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import argparse

from src.model import ModelFactory
from src.data_preprocessing import MultiPointPreprocessor
from src.logger import TrainingLogger
from src.visualization import plot_sequence_prediction_comparison, plot_predictions
from src.clustering import TimeSeriesClusterer
from src.gpu_config import get_model_config, setup_gpu_memory
from src.quadtree_clustering import QuadTreeClusterer, quadtree_clustering

# Configure GPU memory before any other TensorFlow operations
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

class MultiPointTrainer:
    """Handles training and evaluation of multi-point time series models."""
    
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
        preprocessor: Optional[MultiPointPreprocessor] = None,
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
            model_type: Type of model to use ('lstm', 'conv_autoencoder', 'dense_autoencoder', or 'ml')
            clusterer: Optional clusterer instance to use for point selection
            min_correlation: Minimum correlation threshold for point selection
            max_points: Maximum number of points per leaf in quadtree clustering
            max_depth: Maximum depth of quadtree clustering
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
        self.preprocessor = preprocessor
        
        # Initialize logger
        self.logger = TrainingLogger(
            name="multi_point_trainer",
            log_dir=log_dir
        )
        
        
        # Disable TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
    def _create_model_and_train(self, **params) -> float:
        """
        Create and train model with given parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Validation RMSE score
        """
        # Przygotuj konfigurację modelu w zależności od typu
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
                'use_residual_connections': params['use_residual_connections']
            }
        elif self.model_type == 'ml':
            config = {
                'num_points': self.num_points,
                'sequence_length': self.sequence_length,
                'num_features': self.num_features,
                'output_sequence_length': self.output_sequence_length,
                'learning_rate': params.get('learning_rate', 0.1),
                'max_depth': params.get('max_depth', 5),
                'n_estimators': params.get('n_estimators', 100)
            }
        elif self.model_type == 'conv_autoencoder':
            # Convert string parameters to tuples
            conv_filters_str = params['conv_filters']
            kernel_sizes_str = params['kernel_sizes']
            dropout_rates_str = params['dropout_rates']
            
            # Convert strings to tuples of appropriate types
            conv_filters_tuple = tuple(int(x) for x in conv_filters_str.split('_'))
            kernel_sizes_tuple = tuple(int(x) for x in kernel_sizes_str.split('_'))
            dropout_rates_tuple = tuple(float(x) for x in dropout_rates_str.split('_'))
            
            # Ensure lengths match
            min_length = min(len(conv_filters_tuple), len(kernel_sizes_tuple))
            conv_filters_tuple = conv_filters_tuple[:min_length]
            kernel_sizes_tuple = kernel_sizes_tuple[:min_length]
            
            # Calculate required padding for CNN
            num_conv_layers = len(conv_filters_tuple)
            pool_size = 2
            total_pooling = pool_size ** num_conv_layers
            padded_sequence_length = ((self.sequence_length + total_pooling - 1) // total_pooling) * total_pooling
            
            config = {
                'sequence_length': padded_sequence_length,
                'num_features': 1,
                'output_sequence_length': self.output_sequence_length,
                'filters': conv_filters_tuple,
                'kernel_sizes': kernel_sizes_tuple,
                'pool_sizes': tuple(pool_size for _ in range(num_conv_layers)),
                'dense_units': (128, 64),  # Increased dense units
                'dropout_rate': dropout_rates_tuple[0],
                'learning_rate': params['learning_rate']
            }
        elif self.model_type == 'dense_autoencoder':
            # Convert string parameters to tuples
            encoder_units_str = params['encoder_units']
            decoder_units_str = params['decoder_units']
            dropout_rates_str = params['dropout_rates']
            
            # Convert strings to tuples of appropriate types
            encoder_units = tuple(int(x) for x in encoder_units_str.split('_'))
            decoder_units = tuple(int(x) for x in decoder_units_str.split('_'))
            dropout_rates = tuple(float(x) for x in dropout_rates_str.split('_'))
            
            # Ensure we have enough dropout rates
            total_layers = len(encoder_units) + len(decoder_units)
            if len(dropout_rates) < total_layers:
                dropout_rates = tuple(dropout_rates[0] for _ in range(total_layers))
            
            # Use LeakyReLU for better gradient flow
            activation_functions = tuple('leaky_relu' for _ in range(total_layers))
            
            config = {
                'input_shape': (self.sequence_length, 1),  # Keep temporal dimension
                'output_shape': (self.output_sequence_length,),
                'layer_units': encoder_units + decoder_units,
                'dropout_rates': dropout_rates,
                'activation_functions': activation_functions,
                'output_activation': 'linear',  # Linear activation for regression
                'learning_rate': params['learning_rate']
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Create model using factory
        model = ModelFactory.create_model(self.model_type, config)
        
        # Train model
        if self.model_type == 'lstm':
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min'
            )
            
            history = model.train(
                self.x_train,
                self.y_train,
                self.indices_train,
                params['batch_size'],
                params['epochs'],
                validation_data=(self.x_val, self.y_val, self.indices_val),
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            history = model.train(
                self.x_train,
                self.y_train,
                self.indices_train,
                params.get('batch_size', 32),
                params.get('epochs', 100),
                validation_data=(self.x_val, self.y_val, self.indices_val),
                verbose=0
            )
        
        # Get validation predictions
        val_predictions = model.predict(
            x=self.x_val,
            point_indices=self.indices_val,
            batch_size=params.get('batch_size', 32),
            verbose=0
        )
        
        # Check for invalid values in predictions
        if np.any(np.isnan(val_predictions)) or np.any(np.isinf(val_predictions)):
            self.logger.warning(f"Invalid predictions found with parameters: {params}")
            return 1e6
        
        # Calculate RMSE
        val_rmse = np.sqrt(np.mean((self.y_val - val_predictions) ** 2))
        
        # Check if RMSE is valid
        if np.isnan(val_rmse) or np.isinf(val_rmse) or val_rmse > 1e6:
            self.logger.warning(f"Invalid RMSE ({val_rmse}) with parameters: {params}")
            return 1e6
        print(f"RMSE: {val_rmse}")
        print(f"Params: {params}")
        print(f"Model: {model}")
        print(f"History: {history}")
        # Store if best so far
        if self.best_metrics is None or val_rmse < self.best_metrics['RMSE']:
            self.model = model
            self.best_params = params
            self.best_metrics = {'RMSE': val_rmse}
            self.logger.info(f"New best RMSE: {val_rmse:.4f}")
        
        return float(val_rmse)
            
        #except Exception as e:
        ##   self.logger.error(f"Error in _create_model_and_train: {str(e)}", exc_info=True)
        #    return 1e6

    def _optimize_hyperparameters(self):
        """Optimize hyperparameters using Bayesian optimization."""
        # Define the parameter space based on model type
        if self.model_type == 'lstm':
            dimensions = [
                Categorical(['64_32', '128_64', '256_128', '512_256'], name='lstm_units'),
                Categorical(['0.2_0.2', '0.3_0.3', '0.4_0.4', '0.5_0.5'], name='dropout_rates'),
                Categorical(['32_16', '64_32', '128_64'], name='dense_units'),
                Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
                Categorical([256, 512, 1024, 2048, 4096, 8192], name='batch_size'),
                Categorical([50, 100, 150, 200], name='epochs'),
                Integer(4, 16, name='point_embedding_dim'),
                Categorical([True], name='use_point_embeddings'),
                Categorical([True], name='use_residual_connections')
            ]
        elif self.model_type == 'ml':
            dimensions = [
                Real(0.01, 0.3, name='learning_rate'),
                Integer(3, 10, name='max_depth'),
                Integer(50, 200, name='n_estimators')
            ]
        elif self.model_type == 'conv_autoencoder':
            dimensions = [
                Categorical(['32_64_128', '64_128_256', '128_256_512'], name='conv_filters'),
                Categorical(['3_3_3', '5_5_5', '7_7_7'], name='kernel_sizes'),
                Categorical(['64', '128', '256'], name='dense_units'),
                Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
                Categorical([256, 512, 1024, 2048, 4096, 8192], name='batch_size'),
                Categorical([50, 100, 150, 200], name='epochs'),
                Categorical(['0.1_0.1_0.1', '0.2_0.2_0.2', '0.3_0.3_0.3'], name='dropout_rates'),
                Integer(4, 16, name='latent_dim')
            ]
        elif self.model_type == 'dense_autoencoder':
            dimensions = [
                Categorical(['128_64', '256_128', '512_256'], name='encoder_units'),
                Categorical(['64_128', '128_256', '256_512'], name='decoder_units'),
                Categorical(['0.2_0.2', '0.3_0.3', '0.4_0.4'], name='dropout_rates'),
                Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
                Categorical([256, 512, 1024, 2048, 4096, 8192], name='batch_size'),
                Categorical([50, 100, 150], name='epochs')
            ]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        
        # Create optimizer
        optimizer = Optimizer(
            dimensions=dimensions,
            random_state=42,
            base_estimator='GP',
            n_initial_points=5,
            acq_func='EI'  # Expected Improvement
        )
        
        # Run optimization
        self.logger.info("Starting Bayesian optimization...")
        best_params = None
        best_score = float('inf')
        
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
                elif self.model_type == 'ml':
                    params = {
                        'learning_rate': x[0],
                        'max_depth': x[1],
                        'n_estimators': x[2]
                    }
                elif self.model_type == 'conv_autoencoder':
                    params = {
                        'conv_filters': x[0],
                        'kernel_sizes': x[1],
                        'dense_units': x[2],
                        'learning_rate': x[3],
                        'batch_size': x[4],
                        'epochs': x[5],
                        'dropout_rates': x[6],
                        'latent_dim': x[7]
                    }
                else:  # dense_autoencoder
                    params = {
                        'encoder_units': x[0],
                        'decoder_units': x[1],
                        'dropout_rates': x[2],
                        'learning_rate': x[3],
                        'batch_size': x[4],
                        'epochs': x[5]
                    }
                
                # Evaluate parameters
                score = self._create_model_and_train(**params)

                # Ensure score is valid
                try:
                    if isinstance(score, np.ndarray):
                        if np.isnan(score).any() or np.isinf(score).any():
                            score = 1e6
                    else:
                        score = float(score)
                        if np.isnan(score) or np.isinf(score):
                            score = 1e6
                except (TypeError, ValueError):
                    score = 1e6
                
                # Update optimizer
                optimizer.tell(x, float(score))
                
                # Update progress
                if score < best_score:
                    best_score = score
                    best_params = params
                    self.model = self.model  # Store the best model
                    pbar.set_postfix({'Best RMSE': f"{best_score:.4f}"})
                pbar.update(1)
        
        # Store the final best model and parameters
        self.best_params = best_params
        self.best_metrics = {'RMSE': best_score}
        
        return best_params, best_score
    
    def train_and_evaluate(self, preprocessed_data: Dict[str, Any] = None) -> Tuple[Dict, Dict, pd.DataFrame, Any]:
        """
        Train model and evaluate performance using Bayesian optimization.
        
        Args:
            preprocessed_data: Dictionary containing preprocessed data from DataPreprocessor
            
        Returns:
            Tuple of (best_params, best_metrics, predictions_df, trained_model)
        """
        try:
            if preprocessed_data is None:
                raise ValueError("preprocessed_data must be provided")

            # Extract preprocessed data
            self.x_train = preprocessed_data['x_train']
            self.x_val = preprocessed_data['x_val']
            self.y_train = preprocessed_data['y_train']
            self.y_val = preprocessed_data['y_val']
            self.indices_train = preprocessed_data['indices_train']
            self.indices_val = preprocessed_data['indices_val']
            self.num_points = preprocessed_data['num_points']
            self.num_features = preprocessed_data['num_features']
            
            # Store all point columns for later use
            self.all_point_columns = preprocessed_data['all_point_columns']
            
            # Run Bayesian optimization
            best_params, best_score = self._optimize_hyperparameters()
            
            return self.best_params, None, None, self.model
            
        except Exception as e:
            self.logger.error("Error in train_and_evaluate", exc_info=True)
            raise

    def generate_predictions_for_all_test_points(self, generate_plots: bool = False) -> pd.DataFrame:
        """
        Generate predictions for all points in the test set using the trained model.
        
        Args:
            generate_plots: Whether to generate prediction plots (default: True)
            
        Returns:
            DataFrame with predictions for all points
        """
        try:
            # Store the number of points used during training
            training_num_points = len(self.preprocessor.point_columns)
            all_points = self.all_point_columns
            
            # Create predictions dictionary for all points
            all_predictions = {point: {} for point in all_points}
            
            # Get the last sequence_length rows from training data for initial context
            last_train_data = self.train_data.iloc[-self.sequence_length:]
            # Concatenate with test data to ensure we have historical context
            full_test_data = pd.concat([last_train_data, self.test_data])
            
            # Process points in batches of the same size as during training
            for start_idx in tqdm(range(0, len(all_points), training_num_points), desc="Generating predictions"):
                end_idx = min(start_idx + training_num_points, len(all_points))
                current_points = all_points[start_idx:end_idx]
                
                # If this is the last batch and it's smaller than training_num_points,
                # pad it with some points we already processed
                if len(current_points) < training_num_points:
                    padding_needed = training_num_points - len(current_points)
                    current_points = current_points + all_points[:padding_needed]
                
                # Set preprocessor to use current batch of points
                self.preprocessor.point_columns = current_points
                
                # Preprocess test data with current points
                sequences, targets, indices = self.preprocessor.transform(full_test_data)
                
                # Ensure proper shapes for model input
                if self.model_type == 'conv_autoencoder':
                    if len(sequences.shape) == 2:
                        sequences = sequences.reshape(-1, self.sequence_length, 1)
                elif self.model_type == 'lstm':
                    if len(sequences.shape) == 2:
                        sequences = sequences.reshape(-1, self.sequence_length, 1)
                    if len(indices.shape) == 1:
                        indices = indices.reshape(-1, 1)
                
                # Generate predictions
                if self.model_type == 'lstm':
                    predictions = self.model.predict(
                        x=sequences,
                        point_indices=indices,
                        batch_size=self.best_params['batch_size'],
                        verbose=0
                    )
                else:  # For conv_autoencoder and ML models
                    predictions = self.model.predict(
                        x=sequences,
                        point_indices=indices,
                        batch_size=32,
                        verbose=0
                    )
                
                # Inverse transform predictions
                predictions_original = self.preprocessor.inverse_transform_values(
                    predictions,
                    indices
                )
                
                num_points = len(current_points)
                num_test_samples = len(sequences) // num_points
                
                # Calculate sequence start dates - now including the overlap period
                sequence_starts = []
                for i in range(0, len(sequences), num_points):
                    sequence_starts.append(full_test_data.index[i // num_points])
                
                # Process predictions for each point in current batch
                for point_idx, point_name in enumerate(current_points):
                    # Skip if this is a padding point and we've already processed it
                    if point_idx >= end_idx - start_idx:
                        continue
                        
                    for seq_idx in range(num_test_samples):
                        pred_idx = point_idx * num_test_samples + seq_idx
                        sequence_predictions = predictions_original[pred_idx]
                        start_date = sequence_starts[seq_idx]
                        
                        for step, pred_value in enumerate(sequence_predictions):
                            step = step * 6
                            target_date = start_date + pd.Timedelta(days=(self.sequence_length * 6 + step))
                            
                            # Only store predictions that fall within the test period
                            if target_date in self.test_data.index:
                                if target_date not in all_predictions[point_name]:
                                    all_predictions[point_name][target_date] = []
                                all_predictions[point_name][target_date].append(pred_value)
            
            # Create predictions DataFrame with test data index
            predictions_df = pd.DataFrame(
                np.nan,
                index=self.test_data.index,
                columns=all_points
            )
            
            # Aggregate predictions
            for point_name in all_points:
                for date, values in all_predictions[point_name].items():
                    predictions_df.loc[date, point_name] = np.mean(values)
            
            # Generate visualizations only if requested
            if generate_plots:
                predictions_viz_dir = os.path.join(self.output_dir, "predictions_plots")
                os.makedirs(predictions_viz_dir, exist_ok=True)
                
                self.logger.info("Generating prediction plots...")
                for point_name in tqdm(all_points, desc="Generating plots", unit="plot"):
                    plot_predictions(
                        actual_data=self.test_data,
                        predicted_data=predictions_df,
                        point_names=[point_name],
                        output_dir=predictions_viz_dir,
                        plot_type='interactive',
                        show_confidence=True
                    )
            
            # Restore original point columns
            self.preprocessor.point_columns = self.all_point_columns
            
            return predictions_df
            
        except Exception as e:
            self.logger.error("Error in generate_predictions_for_all_test_points", exc_info=True)
            raise