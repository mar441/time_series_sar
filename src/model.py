"""
Model module containing the neural network architectures.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Input, Concatenate, Dropout,
    LayerNormalization, Embedding, Reshape, TimeDistributed, Lambda,
    RepeatVector, Conv1D, MaxPooling1D, UpSampling1D, Flatten,
    BatchNormalization, ReLU, LeakyReLU, Activation
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
from abc import ABC, abstractmethod
import numpy as np
import tensorflow.keras.backend as K
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
    mean_directional_accuracy,
    r_squared,
    weighted_mean_absolute_error
)


class ModelFactory:
    """Factory class for creating different types of models."""
    
    @staticmethod
    def create_model(
        model_type: str,
        config: Dict[str, Any]
    ) -> 'TimeSeriesModel':
        """
        Create a model instance based on type and configuration.
        
        Args:
            model_type: Type of model to create ('lstm', 'conv_autoencoder', 'dense_autoencoder', 'ml')
            config: Model configuration dictionary
            
        Returns:
            Instance of TimeSeriesModel
            
        Raises:
            ValueError: If model type is not supported
        """
        if model_type == 'lstm':
            return MultiPointTimeSeriesModel(
                num_points=config['num_points'],
                sequence_length=config['sequence_length'],
                num_features=config['num_features'],
                output_sequence_length=config['output_sequence_length'],
                lstm_units=config.get('lstm_units', (256, 128)),
                dropout_rates=config.get('dropout_rates', (0.2, 0.2)),
                dense_units=config.get('dense_units', (64,)),
                learning_rate=config.get('learning_rate', 0.001),
                point_embedding_dim=config.get('point_embedding_dim', 8),
                use_point_embeddings=config.get('use_point_embeddings', True),
                use_residual_connections=config.get('use_residual_connections', True)
            )
        elif model_type == 'ml':
            return MLModel(
                num_points=config['num_points'],
                sequence_length=config['sequence_length'],
                num_features=config['num_features'],
                output_sequence_length=config['output_sequence_length'],
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', 5),
                n_estimators=config.get('n_estimators', 100)
            )
        elif model_type == 'conv_autoencoder':
            return ConvolutionalAutoencoder(
                sequence_length=config['sequence_length'],
                num_features=config['num_features'],
                output_sequence_length=config['output_sequence_length'],
                filters=config.get('filters', (32, 64, 128)),
                kernel_sizes=config.get('kernel_sizes', (3, 3, 3)),
                pool_sizes=config.get('pool_sizes', (2, 2, 2)),
                dense_units=config.get('dense_units', (64,)),
                dropout_rate=config.get('dropout_rate', 0.2),
                learning_rate=config.get('learning_rate', 0.001)
            )
        elif model_type == 'dense_autoencoder':
            return DenseAutoencoder(
                input_shape=config['input_shape'],
                output_shape=config['output_shape'],
                learning_rate=config['learning_rate'],
                layer_units=config.get('layer_units', (128, 64, 32)),
                dropout_rates=config.get('dropout_rates', (0.2, 0.2, 0.2)),
                activation_functions=config.get('activation_functions', ('relu', 'relu', 'relu')),
                output_activation=config.get('output_activation', 'linear')
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def load_model(
        model_path: str,
        model_type: str
    ) -> 'TimeSeriesModel':
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            model_type: Type of model to load ('lstm', 'conv_autoencoder', 'dense_autoencoder', 'ml')
            
        Returns:
            Loaded model instance
        """
        if model_type == 'lstm':
            return MultiPointTimeSeriesModel.load(model_path, model_type)
        elif model_type == 'conv_autoencoder':
            return ConvolutionalAutoencoder.load(model_path, model_type)
        elif model_type == 'dense_autoencoder':
            return DenseAutoencoder.load(model_path, model_type)
        elif model_type == 'ml':
            return MLModel.load(model_path, model_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    model_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    learning_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'learning_rate': self.learning_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(
            model_type=config_dict['model_type'],
            input_shape=tuple(config_dict['input_shape']),
            output_shape=tuple(config_dict['output_shape']),
            learning_rate=config_dict['learning_rate']
        )


class ModelArchitectureValidator:
    """Validates model architecture configurations."""
    
    @staticmethod
    def validate_lstm_config(config: Dict[str, Any]) -> bool:
        """
        Validate LSTM model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = [
            'num_points', 'sequence_length', 'num_features',
            'output_sequence_length', 'lstm_units', 'dropout_rates',
            'dense_units', 'learning_rate'
        ]
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate shapes and sizes
        if len(config['lstm_units']) != len(config['dropout_rates']):
            raise ValueError("Number of LSTM units must match number of dropout rates")
        
        if config['sequence_length'] <= 0:
            raise ValueError("sequence_length must be positive")
        
        if config['output_sequence_length'] <= 0:
            raise ValueError("output_sequence_length must be positive")
        
        if config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        
        return True
    
    @staticmethod
    def validate_cnn_config(config: Dict[str, Any]) -> bool:
        """
        Validate CNN model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = [
            'sequence_length', 'num_features', 'output_sequence_length',
            'filters', 'kernel_sizes', 'pool_sizes', 'dense_units',
            'dropout_rate', 'learning_rate'
        ]
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate shapes and sizes
        if len(config['filters']) != len(config['kernel_sizes']):
            raise ValueError("Number of filters must match number of kernel sizes")
        
        if len(config['filters']) != len(config['pool_sizes']):
            raise ValueError("Number of filters must match number of pool sizes")
        
        if config['sequence_length'] <= 0:
            raise ValueError("sequence_length must be positive")
        
        if config['output_sequence_length'] <= 0:
            raise ValueError("output_sequence_length must be positive")
        
        if config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        
        if any(f <= 0 for f in config['filters']):
            raise ValueError("All filter sizes must be positive")
        
        if any(k <= 0 for k in config['kernel_sizes']):
            raise ValueError("All kernel sizes must be positive")
        
        if any(p <= 0 for p in config['pool_sizes']):
            raise ValueError("All pool sizes must be positive")
        
        if config['dropout_rate'] < 0 or config['dropout_rate'] > 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        return True
    
    @staticmethod
    def validate_dense_config(config: Dict[str, Any]) -> bool:
        """Validate Dense Autoencoder configuration."""
        required_keys = [
            'input_shape', 'output_shape', 'learning_rate',
            'layer_units', 'dropout_rates', 'activation_functions',
            'output_activation'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        if len(config['layer_units']) != len(config['dropout_rates']):
            raise ValueError("Number of layer units must match number of dropout rates")
        
        return True


class GPUManager:
    """Manages GPU memory and device placement."""
    
    @staticmethod
    def setup_gpu_memory():
        """Set up GPU memory growth to prevent memory allocation issues."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
            except RuntimeError as e:
                logging.error(f"Error setting up GPU memory growth: {str(e)}")
    
    @staticmethod
    def get_available_gpus() -> List[str]:
        """Get list of available GPU devices."""
        return tf.config.list_physical_devices('GPU')
    
    @staticmethod
    def set_gpu_device(device_index: int):
        """Set specific GPU device to use."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[device_index], 'GPU')
                logging.info(f"Set GPU device to index {device_index}")
            except RuntimeError as e:
                logging.error(f"Error setting GPU device: {str(e)}")


class ModelSerializer:
    """Handles model serialization and deserialization."""
    
    @staticmethod
    def save_model(
        model: Union[Model, xgb.XGBRegressor],
        config: ModelConfig,
        save_dir: str,
        model_name: str
    ):
        """
        Save model and its configuration.
        
        Args:
            model: Model to save
            config: Model configuration
            save_dir: Directory to save model
            model_name: Name of the model
        """
        save_path = Path(save_dir) / model_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if isinstance(model, Model):
            model.save(save_path / 'model.h5')
        else:
            model.save_model(save_path / 'model.json')
        
        # Save configuration
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        
        logging.info(f"Model saved to {save_path}")
    
    @staticmethod
    def load_model(
        load_dir: str,
        model_name: str
    ) -> Tuple[Union[Model, xgb.XGBRegressor], ModelConfig]:
        """
        Load model and its configuration.
        
        Args:
            load_dir: Directory to load model from
            model_name: Name of the model
            
        Returns:
            Tuple of (loaded_model, model_config)
        """
        load_path = Path(load_dir) / model_name
        
        # Load configuration
        config_path = load_path / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
        
        # Load model
        if config.model_type in ['lstm', 'conv_autoencoder', 'dense_autoencoder']:
            model = load_model(load_path / 'model.h5')
        else:
            model = xgb.XGBRegressor()
            model.load_model(load_path / 'model.json')
        
        logging.info(f"Model loaded from {load_path}")
        return model, config


class TimeSeriesModel(ABC):
    """Abstract base class for time series models."""
    
    def __init__(self):
        """Initialize model."""
        self.model = None
        self.config = None
        GPUManager.setup_gpu_memory()
    
    @abstractmethod
    def build(self) -> Model:
        """Build and return the model."""
        pass
    
    @abstractmethod
    def train(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        point_indices_train: tf.Tensor,
        batch_size: int,
        epochs: int,
        validation_data: Optional[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(
        self,
        x: tf.Tensor,
        point_indices: tf.Tensor,
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """Make predictions using the model."""
        pass
    
    def save(self, save_dir: str, model_name: str):
        """Save model and configuration."""
        ModelSerializer.save_model(self.model, self.config, save_dir, model_name)
    
    @classmethod
    def load(cls, load_dir: str, model_name: str) -> 'TimeSeriesModel':
        """Load model and configuration."""
        model, config = ModelSerializer.load_model(load_dir, model_name)
        instance = cls()
        instance.model = model
        instance.config = config
        return instance


class MultiPointTimeSeriesModel(TimeSeriesModel):
    """LSTM-based model for multi-point time series forecasting."""
    
    def __init__(
        self,
        num_points: int,
        sequence_length: int,
        num_features: int,
        output_sequence_length: int,
        lstm_units: Tuple[int, ...] = (256, 128),
        dropout_rates: Tuple[float, ...] = (0.2, 0.2),
        dense_units: Tuple[int, ...] = (64,),
        learning_rate: float = 0.001,
        point_embedding_dim: int = 8,
        use_point_embeddings: bool = True,
        use_residual_connections: bool = True
    ):
        """Initialize model with given parameters."""
        super().__init__()
        
        # Validate configuration
        config = {
            'num_points': num_points,
            'sequence_length': sequence_length,
            'num_features': num_features,
            'output_sequence_length': output_sequence_length,
            'lstm_units': lstm_units,
            'dropout_rates': dropout_rates,
            'dense_units': dense_units,
            'learning_rate': learning_rate
        }
        ModelArchitectureValidator.validate_lstm_config(config)
        
        self.num_points = num_points
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.output_sequence_length = output_sequence_length
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.point_embedding_dim = point_embedding_dim
        self.use_point_embeddings = use_point_embeddings
        self.use_residual_connections = use_residual_connections
        
        self.config = ModelConfig(
            model_type='lstm',
            input_shape=(sequence_length, num_features),
            output_shape=(output_sequence_length,),
            learning_rate=learning_rate
        )
        
        self.model = self.build()
    
    def build(self) -> Model:
        """Build and compile the model."""
        # Input layers
        sequence_input = Input(shape=(self.sequence_length, 1), name='sequence_input')
        point_indices = Input(shape=(1,), name='point_indices')
        
        # Point embeddings
        if self.use_point_embeddings:
            point_embedding = Embedding(
                input_dim=self.num_points,
                output_dim=self.point_embedding_dim,
                name='point_embedding'
            )(point_indices)

            point_features = Dense(
                self.point_embedding_dim,
                activation='relu',
                name='point_features'
            )(point_embedding)

            point_embedding_reshaped = Lambda(
                lambda x: tf.repeat(x, repeats=self.sequence_length, axis=1),
                output_shape=lambda input_shape: (input_shape[0], self.sequence_length, input_shape[-1])
            )(point_features)
            
            x = Concatenate(axis=2)([sequence_input, point_embedding_reshaped])
        else:
            x = sequence_input
        
        # LSTM layers with residual connections
        lstm_output = x
        for i, (units, dropout_rate) in enumerate(zip(self.lstm_units, self.dropout_rates)):
            lstm_layer = LSTM(
                units=units,
                return_sequences=True,
                name=f'lstm_{i}'
            )(lstm_output)
            
            lstm_output = Dropout(dropout_rate, name=f'dropout_{i}')(lstm_layer)
            lstm_output = LayerNormalization(name=f'layer_norm_{i}')(lstm_output)
            
            if self.use_residual_connections and i > 0:
                if lstm_output.shape[-1] != x.shape[-1]:
                    x = Dense(units, name=f'residual_projection_{i}')(x)
                lstm_output = tf.keras.layers.Add(name=f'residual_{i}')([x, lstm_output])
            
            x = lstm_output
        
        # Dense layers for output
        for i, units in enumerate(self.dense_units):
            x = TimeDistributed(Dense(units, activation='relu'), name=f'dense_{i}')(x)
        
        # Final output layer
        outputs = TimeDistributed(Dense(1), name='output')(x)
        outputs = Lambda(
            lambda x: tf.squeeze(x[:, -self.output_sequence_length:, :], axis=-1),
            name='reshape_output'
        )(outputs)
        
        # Create and compile model
        model = Model(inputs=[sequence_input, point_indices], outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=[
                'mae',
                root_mean_squared_error,
                mean_absolute_percentage_error,
                symmetric_mean_absolute_percentage_error,
                mean_directional_accuracy
            ]
        )
        
        return model
    
    def train(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        point_indices_train: tf.Tensor,
        batch_size: int,
        epochs: int,
        validation_data: Optional[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """Train the model."""
        train_data = {
            'sequence_input': x_train,
            'point_indices': point_indices_train
        }
        
        if validation_data is not None:
            x_val, y_val, point_indices_val = validation_data
            validation_data = (
                {
                    'sequence_input': x_val,
                    'point_indices': point_indices_val
                },
                y_val
            )
        
        return self.model.fit(
            train_data,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

    def predict(
        self,
        x: tf.Tensor,
        point_indices: tf.Tensor,
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """Make predictions using the model."""
        # Ensure point_indices has correct shape
        if len(point_indices.shape) == 1:
            point_indices = point_indices.reshape(-1, 1)
        
        # Ensure x has correct shape (batch_size, sequence_length, 1)
        if len(x.shape) == 2:
            x = x.reshape(-1, x.shape[1], 1)
            
        return self.model.predict(
            {
                'sequence_input': x,
                'point_indices': point_indices
            },
            batch_size=batch_size,
            verbose=verbose
        )


class ConvolutionalAutoencoder(TimeSeriesModel):
    """Convolutional autoencoder for time series forecasting."""
    
    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        output_sequence_length: int,
        filters: Tuple[int, ...] = (32, 64, 128),
        kernel_sizes: Tuple[int, ...] = (3, 3, 3),
        pool_sizes: Tuple[int, ...] = (2, 2, 2),
        dense_units: Tuple[int, ...] = (64,),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """Initialize CNN autoencoder."""
        super().__init__()
        
        # Calculate required padding for sequence length
        total_pooling = 1
        for pool_size in pool_sizes:
            total_pooling *= pool_size
            
        # Pad sequence_length to be divisible by total pooling factor
        self.padded_sequence_length = sequence_length
        if sequence_length % total_pooling != 0:
            self.padded_sequence_length = ((sequence_length + total_pooling - 1) // total_pooling) * total_pooling
            
        # Store original sequence length for later use
        self.original_sequence_length = sequence_length
        
        # Validate configuration
        config = {
            'sequence_length': self.padded_sequence_length,  # Use padded length
            'num_features': num_features,
            'output_sequence_length': output_sequence_length,
            'filters': filters,
            'kernel_sizes': kernel_sizes,
            'pool_sizes': pool_sizes,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }
        ModelArchitectureValidator.validate_cnn_config(config)
        
        self.sequence_length = self.padded_sequence_length  # Use padded length
        self.num_features = num_features
        self.output_sequence_length = output_sequence_length
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Calculate the size after pooling operations
        self.final_length = self.padded_sequence_length
        for pool_size in pool_sizes:
            self.final_length = self.final_length // pool_size
        self.flattened_size = self.final_length * filters[-1]
        
        self.config = ModelConfig(
            model_type='conv_autoencoder',
            input_shape=(self.padded_sequence_length, num_features),
            output_shape=(output_sequence_length,),
            learning_rate=learning_rate
        )
        
        self.model = self.build()
    
    def _pad_sequence(self, x: tf.Tensor) -> tf.Tensor:
        """Pad sequence to match required length."""
        if len(x.shape) == 2:
            x = x.reshape(-1, x.shape[1], 1)
            
        if x.shape[1] < self.padded_sequence_length:
            padding = self.padded_sequence_length - x.shape[1]
            return np.pad(x, ((0, 0), (0, padding), (0, 0)), mode='edge')
        return x
    
    def train(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        point_indices_train: tf.Tensor,
        batch_size: int,
        epochs: int,
        validation_data: Optional[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """Train the model."""
        if validation_data is not None:
            x_val, y_val, _ = validation_data
            validation_data = (x_val, y_val)
        
        # Pad sequences if needed
        x_train = self._pad_sequence(x_train)
        if validation_data is not None:
            x_val = self._pad_sequence(x_val)
            validation_data = (x_val, y_val)
        
        return self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

    def predict(
        self,
        x: tf.Tensor,
        point_indices: tf.Tensor,
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """Make predictions using the model."""
        # Pad sequence if needed
        x = self._pad_sequence(x)
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def build(self) -> Model:
        """Build and compile the model."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.num_features))
        
        # Encoder
        x = inputs
        for i, (filters, kernel_size, pool_size) in enumerate(zip(
            self.filters, self.kernel_sizes, self.pool_sizes
        )):
            x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                name=f'conv_{i}'
            )(x)
            x = MaxPooling1D(pool_size=pool_size, name=f'pool_{i}')(x)
            x = BatchNormalization(name=f'batch_norm_{i}')(x)
            x = Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        # Flatten layer with explicit shape
        x = Flatten(name='flatten')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = Dense(units, activation='relu', name=f'dense_{i}')(x)
            x = Dropout(self.dropout_rate, name=f'dense_dropout_{i}')(x)
        
        # Final dense layer for output
        outputs = Dense(self.output_sequence_length, activation='linear', name='output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model


class DenseAutoencoder(TimeSeriesModel):
    """Dense autoencoder for time series forecasting."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        learning_rate: float = 0.001,
        layer_units: Tuple[int, ...] = (128, 64, 32),
        dropout_rates: Tuple[float, ...] = (0.2, 0.2, 0.2),
        activation_functions: Tuple[str, ...] = ('relu', 'relu', 'relu'),
        output_activation: str = 'linear'
    ):
        """Initialize dense autoencoder."""
        super().__init__()
        
        # Validate configuration
        config = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'learning_rate': learning_rate,
            'layer_units': layer_units,
            'dropout_rates': dropout_rates,
            'activation_functions': activation_functions,
            'output_activation': output_activation
        }
        ModelArchitectureValidator.validate_dense_config(config)
        
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.dropout_rates = dropout_rates
        self.activation_functions = activation_functions
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.output_shape = output_shape

        self.config = ModelConfig(
            model_type='dense_autoencoder',
            input_shape=input_shape,
            output_shape=output_shape,
            learning_rate=learning_rate
        )
        
        self.model = self.build()
    
    def _build_encoder(self, inputs):
        """Build encoder part of the model."""
        x = inputs
        
        # Flatten input while keeping batch size
        x = Reshape((-1,))(x)
        
        # Encoder layers
        for units, rate, activation in zip(
            self.layer_units,
            self.dropout_rates,
            self.activation_functions
        ):
            x = Dense(units)(x)
            x = BatchNormalization()(x)
            if activation == 'relu':
                x = ReLU()(x)
            elif activation == 'leaky_relu':
                x = LeakyReLU(alpha=0.1)(x)
            else:
                x = Activation(activation)(x)
            x = Dropout(rate)(x)
        
        return x
    
    def _build_decoder(self, encoded):
        """Build decoder part of the model."""
        x = encoded
        
        # Decoder layers (reverse of encoder)
        for units, rate, activation in zip(
            reversed(self.layer_units[:-1]),  # Skip last encoder layer
            reversed(self.dropout_rates[:-1]),
            reversed(self.activation_functions[:-1])
        ):
            x = Dense(units)(x)
            x = BatchNormalization()(x)
            if activation == 'relu':
                x = ReLU()(x)
            elif activation == 'leaky_relu':
                x = LeakyReLU(alpha=0.1)(x)
            else:
                x = Activation(activation)(x)
            x = Dropout(rate)(x)
        
        # Final output layer
        x = Dense(np.prod(self.output_shape))(x)
        x = Activation(self.output_activation)(x)
        outputs = Reshape(self.output_shape)(x)
        
        return outputs

    def build(self) -> Model:
        """Build and compile the model."""
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Build encoder-decoder
        encoded = self._build_encoder(inputs)
        outputs = self._build_decoder(encoded)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=[
                'mae',
                root_mean_squared_error,
                mean_absolute_percentage_error,
                symmetric_mean_absolute_percentage_error,
                mean_directional_accuracy
            ]
        )
        
        return model
    
    def train(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        point_indices_train: tf.Tensor,
        batch_size: int,
        epochs: int,
        validation_data: Optional[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """Train the model."""
        if validation_data is not None:
            x_val, y_val, _ = validation_data
            validation_data = (x_val, y_val)
        
        return self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

    def predict(
        self,
        x: tf.Tensor,
        point_indices: tf.Tensor,
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """Make predictions using the model."""
        # Ensure input has correct shape
        if len(x.shape) == 2:
            x = x.reshape(-1, *self.input_shape)
        elif len(x.shape) == 3:
            x = x.reshape(-1, *self.input_shape)
            
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)


class MLModel(TimeSeriesModel):
    """Machine learning model for time series forecasting using XGBoost."""
    
    def __init__(
        self,
        num_points: int,
        sequence_length: int,
        num_features: int,
        output_sequence_length: int,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        n_estimators: int = 100
    ):
        """Initialize ML model."""
        super().__init__()
        
        self.num_points = num_points
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.output_sequence_length = output_sequence_length
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        
        self.config = ModelConfig(
            model_type='ml',
            input_shape=(sequence_length * num_features,),
            output_shape=(output_sequence_length,),
            learning_rate=learning_rate
        )
        
        self.model = self.build()
    
    def build(self) -> xgb.XGBRegressor:
        """Build the XGBoost model."""
        model = MultiOutputRegressor(
            xgb.XGBRegressor(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                objective='reg:squarederror',
                n_jobs=-1
            )
        )
        return model
    
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        point_indices_train: np.ndarray,
        batch_size: int,
        epochs: int,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List[Any]] = None,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """Train the model."""
        # Reshape input to 2D for XGBoost
        x_train_2d = x_train.reshape(x_train.shape[0], -1)
        
        # Train model
        self.model.fit(x_train_2d, y_train)
        
        # Create history dict to match Keras style
        history = {
            'loss': [0.0],  # Placeholder since XGBoost doesn't provide epoch-wise loss
            'val_loss': [0.0] if validation_data is not None else []
        }
        
        return history
    
    def predict(
        self,
        x: np.ndarray,
        point_indices: np.ndarray,
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """Make predictions."""
        # Reshape input to 2D for XGBoost
        x_2d = x.reshape(x.shape[0], -1)
        
        # Make predictions
        predictions = self.model.predict(x_2d)
        
        return predictions