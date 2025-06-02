"""
Model module containing PyTorch neural network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass
import numpy as np
import json
import logging
from pathlib import Path
import os
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics.regression import MeanAbsolutePercentageError


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


class ModelSerializer:
    """Handles model serialization and deserialization."""
    
    @staticmethod
    def save_model(
        model: pl.LightningModule,
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
        torch.save(model.state_dict(), save_path / 'model.pt')
        
        # Save configuration
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        
        logging.info(f"Model saved to {save_path}")
    
    @staticmethod
    def load_model(
        model_class: type,
        load_dir: str,
        model_name: str
    ) -> Tuple[pl.LightningModule, ModelConfig]:
        """
        Load model and its configuration.
        
        Args:
            model_class: Class of the model to load
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
        
        # Create model instance
        model = model_class.load_from_config(config)
        
        # Load model weights
        model.load_state_dict(torch.load(load_path / 'model.pt'))
        
        logging.info(f"Model loaded from {load_path}")
        return model, config


class ModelFactory:
    """Factory class for creating different types of PyTorch models."""
    
    @staticmethod
    def create_model(
        model_type: str,
        config: Dict[str, Any]
    ) -> pl.LightningModule:
        """
        Create a model instance based on type and configuration.
        
        Args:
            model_type: Type of model to create ('lstm', 'conv_autoencoder', 'dense_autoencoder')
            config: Model configuration dictionary
            
        Returns:
            Instance of LightningModule
            
        Raises:
            ValueError: If model type is not supported
        """
        if model_type == 'lstm':
            return MultiPointTimeSeriesModelPL(
                num_points=config['num_points'],
                sequence_length=config['sequence_length'],
                num_features=config['num_features'],
                output_sequence_length=config['output_sequence_length'],
                lstm_units=config.get('lstm_units', (64, 32)),
                dropout_rates=config.get('dropout_rates', (0.2, 0.2)),
                dense_units=config.get('dense_units', (32,)),
                learning_rate=config.get('learning_rate', 0.001),
                point_embedding_dim=config.get('point_embedding_dim', 8),
                use_point_embeddings=config.get('use_point_embeddings', True),
                use_residual_connections=config.get('use_residual_connections', True),
                weight_decay=config.get('weight_decay', 1e-5),
                lr_scheduler_factor=config.get('lr_scheduler_factor', 0.5),
                lr_scheduler_patience=config.get('lr_scheduler_patience', 10)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def load_model(
        model_path: str,
        model_type: str
    ) -> pl.LightningModule:
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            model_type: Type of model to load ('lstm', 'conv_autoencoder', 'dense_autoencoder')
            
        Returns:
            Loaded model instance
        """
        if model_type == 'lstm':
            return MultiPointTimeSeriesModelPL.load(model_path, model_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class MultiPointTimeSeriesModelPL(pl.LightningModule):
    """LSTM-based model for multi-point time series forecasting using PyTorch Lightning."""
    
    def __init__(
        self,
        num_points: int,
        sequence_length: int,
        num_features: int,
        output_sequence_length: int,
        lstm_units: Tuple[int, ...],
        dropout_rates: Tuple[float, ...],
        dense_units: Tuple[int, ...],
        learning_rate: float = 0.001,
        point_embedding_dim: int = 8,
        use_point_embeddings: bool = True,
        use_residual_connections: bool = True,
        weight_decay: float = 1e-5,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10
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
        
        self.save_hyperparameters()
        
        # Store configuration parameters
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
        self.weight_decay = weight_decay
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        
        # Build model components
        if self.use_point_embeddings:
            self.point_embedding = nn.Embedding(
                num_embeddings=num_points,
                embedding_dim=point_embedding_dim
            )
            self.point_features = nn.Linear(
                point_embedding_dim,
                point_embedding_dim
            )
            # Input to LSTM will have additional embedding dimensions
            lstm_input_size = num_features + point_embedding_dim
        else:
            lstm_input_size = num_features
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_units[0],
                batch_first=True,
                num_layers=1
            )
        )
        self.layer_norms.append(nn.LayerNorm(lstm_units[0]))
        self.dropouts.append(nn.Dropout(dropout_rates[0]))
        self.residual_projections.append(None)  # No residual for first layer
        
        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_units[i-1],
                    hidden_size=lstm_units[i],
                    batch_first=True,
                    num_layers=1
                )
            )
            self.layer_norms.append(nn.LayerNorm(lstm_units[i]))
            self.dropouts.append(nn.Dropout(dropout_rates[i]))
            
            # Add residual projection if needed
            if use_residual_connections:
                if lstm_units[i-1] != lstm_units[i]:
                    self.residual_projections.append(
                        nn.Linear(lstm_units[i-1], lstm_units[i])
                    )
                else:
                    self.residual_projections.append(None)
            else:
                self.residual_projections.append(None)
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        current_size = lstm_units[-1]
        
        for units in dense_units:
            self.dense_layers.append(nn.Linear(current_size, units))
            current_size = units
        
        # Output layer
        self.output_layer = nn.Linear(current_size, 1)
        
        # Metrics
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.train_r2 = R2Score()
        self.train_mape = MeanAbsolutePercentageError()
        
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.val_mape = MeanAbsolutePercentageError()
        
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        self.test_mape = MeanAbsolutePercentageError()
        
        # Config for saving/loading
        self.config = ModelConfig(
            model_type='lstm',
            input_shape=(sequence_length, num_features),
            output_shape=(output_sequence_length,),
            learning_rate=learning_rate
        )
    
    def forward(self, sequence_input, point_indices):
        """Forward pass through the model."""
        batch_size = sequence_input.size(0)
        
        logging.info(f"Forward pass - Input shapes - sequence_input: {sequence_input.shape}, point_indices: {point_indices.shape}")
        logging.info(f"Forward pass - Input contiguity - sequence_input: {sequence_input.is_contiguous()}, point_indices: {point_indices.is_contiguous()}")
        
        # Ensure input tensors are contiguous
        sequence_input = sequence_input.contiguous()
        point_indices = point_indices.contiguous()
        
        logging.info(f"After contiguous() - sequence_input: {sequence_input.is_contiguous()}, point_indices: {point_indices.is_contiguous()}")
        
        # Process point embeddings if used
        if self.use_point_embeddings:
            # Get point embeddings
            point_embedded = self.point_embedding(point_indices)  # [batch_size, 1, embedding_dim]
            logging.info(f"After embedding - point_embedded shape: {point_embedded.shape}, contiguous: {point_embedded.is_contiguous()}")
            
            point_embedded = point_embedded.squeeze(1)  # [batch_size, embedding_dim]
            logging.info(f"After squeeze - point_embedded shape: {point_embedded.shape}, contiguous: {point_embedded.is_contiguous()}")
            
            # Project embeddings
            point_features = self.point_features(point_embedded)  # [batch_size, embedding_dim]
            point_features = F.relu(point_features)
            logging.info(f"After projection - point_features shape: {point_features.shape}, contiguous: {point_features.is_contiguous()}")
            
            # Repeat embedding for each time step
            point_features = point_features.unsqueeze(1).repeat(1, self.sequence_length, 1)  # [batch_size, seq_len, embedding_dim]
            logging.info(f"After repeat - point_features shape: {point_features.shape}, contiguous: {point_features.is_contiguous()}")
            
            # Concatenate with sequence input
            x = torch.cat([sequence_input, point_features], dim=2)  # [batch_size, seq_len, num_features + embedding_dim]
            logging.info(f"After concatenation - x shape: {x.shape}, contiguous: {x.is_contiguous()}")
        else:
            x = sequence_input
        
        # Process through LSTM layers with residual connections
        for i, (lstm, layer_norm, dropout, residual_proj) in enumerate(
            zip(self.lstm_layers, self.layer_norms, self.dropouts, self.residual_projections)
        ):
            # Store input for residual connection
            residual = x
            
            # Apply LSTM
            lstm_out, _ = lstm(x)
            x = lstm_out
            logging.info(f"After LSTM {i} - x shape: {x.shape}, contiguous: {x.is_contiguous()}")
            
            # Apply dropout and layer normalization
            x = dropout(x)
            x = layer_norm(x)
            
            # Apply residual connection if not the first layer and residual connections are enabled
            if i > 0 and self.use_residual_connections:
                if residual_proj is not None:
                    residual = residual_proj(residual)
                x = x + residual
                logging.info(f"After residual {i} - x shape: {x.shape}, contiguous: {x.is_contiguous()}")
        
        # Process through dense layers
        for i, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
            x = F.relu(x)
            logging.info(f"After dense {i} - x shape: {x.shape}, contiguous: {x.is_contiguous()}")
        
        # Apply output layer to each time step
        x = self.output_layer(x)  # [batch_size, seq_len, 1]
        logging.info(f"After output layer - x shape: {x.shape}, contiguous: {x.is_contiguous()}")
        
        # Take only the required output sequence length from the end
        x = x[:, -self.output_sequence_length:, 0]  # [batch_size, output_seq_len]
        logging.info(f"Final output shape: {x.shape}, contiguous: {x.is_contiguous()}")
        
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, point_indices, y = batch
        y_hat = self(x, point_indices)
        loss = F.mse_loss(y_hat, y)
        
        # Ensure tensors are contiguous before passing to metrics
        y_hat = y_hat.contiguous()
        y = y.contiguous()
        
        # Update and log metrics
        self.train_mse(y_hat, y)
        self.train_mae(y_hat, y)
        self.train_r2(y_hat, y)
        
        # MAPE can have NaN issues if y has zeros, so handle it separately
        try:
            self.train_mape(y_hat, y)
            self.log('train_mape', self.train_mape, on_epoch=True)
        except:
            pass
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', self.train_mse, on_epoch=True)
        self.log('train_mae', self.train_mae, on_epoch=True)
        self.log('train_r2', self.train_r2, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, point_indices, y = batch
        y_hat = self(x, point_indices)
        loss = F.mse_loss(y_hat, y)
        
        # Ensure tensors are contiguous before passing to metrics
        y_hat = y_hat.contiguous()
        y = y.contiguous()
        
        # Update and log metrics
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        self.val_r2(y_hat, y)
        
        # MAPE can have NaN issues if y has zeros, so handle it separately
        try:
            self.val_mape(y_hat, y)
            self.log('val_mape', self.val_mape, on_epoch=True)
        except:
            pass
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', self.val_mse, on_epoch=True)
        self.log('val_mae', self.val_mae, on_epoch=True)
        self.log('val_r2', self.val_r2, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, point_indices, y = batch
        y_hat = self(x, point_indices)
        loss = F.mse_loss(y_hat, y)
        
        # Ensure tensors are contiguous before passing to metrics
        y_hat = y_hat.contiguous()
        y = y.contiguous()
        
        # Update and log metrics
        self.test_mse(y_hat, y)
        self.test_mae(y_hat, y)
        self.test_r2(y_hat, y)
        
        # MAPE can have NaN issues if y has zeros, so handle it separately
        try:
            self.test_mape(y_hat, y)
            self.log('test_mape', self.test_mape)
        except:
            pass
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_mse', self.test_mse)
        self.log('test_mae', self.test_mae)
        self.log('test_r2', self.test_r2)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        print(len(batch))
        if len(batch) == 3:
            x, point_indices, y = batch
        else:
            x, point_indices = batch
        return self(x, point_indices)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def save(self, save_dir: str, model_name: str):
        """Save model and configuration."""
        ModelSerializer.save_model(self, self.config, save_dir, model_name)
    
    @classmethod
    def load(cls, load_dir: str, model_name: str) -> 'MultiPointTimeSeriesModelPL':
        """Load model and configuration."""
        model, _ = ModelSerializer.load_model(cls, load_dir, model_name)
        return model
    
    @classmethod
    def load_from_config(cls, config: ModelConfig) -> 'MultiPointTimeSeriesModelPL':
        """Create model instance from configuration."""
        # Extract hyperparameters from saved config and create a new instance
        if config.model_type != 'lstm':
            raise ValueError(f"Expected model type 'lstm', got {config.model_type}")
        
        # In a real implementation, you would need to store all hyperparameters in the config
        # This is a simplified example assuming the config contains these fields
        model = cls(
            num_points=config.hparams.get('num_points'),
            sequence_length=config.hparams.get('sequence_length'),
            num_features=config.hparams.get('num_features'),
            output_sequence_length=config.hparams.get('output_sequence_length'),
            lstm_units=config.hparams.get('lstm_units'),
            dropout_rates=config.hparams.get('dropout_rates'),
            dense_units=config.hparams.get('dense_units'),
            learning_rate=config.hparams.get('learning_rate', 0.001),
            point_embedding_dim=config.hparams.get('point_embedding_dim', 8),
            use_point_embeddings=config.hparams.get('use_point_embeddings', True),
            use_residual_connections=config.hparams.get('use_residual_connections', True),
            weight_decay=config.hparams.get('weight_decay', 1e-5),
            lr_scheduler_factor=config.hparams.get('lr_scheduler_factor', 0.5),
            lr_scheduler_patience=config.hparams.get('lr_scheduler_patience', 10)
        )
        return model 


class TimeSeriesDataModule(pl.LightningDataModule):
    """Data module for time series data using PyTorch Lightning."""
    
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        point_indices_train: np.ndarray,
        batch_size: int = 32,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        point_indices_val: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None, 
        point_indices_test: Optional[np.ndarray] = None,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """Initialize data module."""
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.point_indices_train = point_indices_train
        self.x_val = x_val
        self.y_val = y_val
        self.point_indices_val = point_indices_val
        self.x_test = x_test
        self.y_test = y_test
        self.point_indices_test = point_indices_test
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
        # Set up datasets immediately
        self.setup()
    
    def setup(self, stage: Optional[str] = None):
        """Set up the datasets."""
        logging.info("Setting up TimeSeriesDataModule...")
        
        # Ensure arrays are numpy arrays and contiguous
        x_train = np.ascontiguousarray(self.x_train)
        y_train = np.ascontiguousarray(self.y_train)
        point_indices_train = np.ascontiguousarray(self.point_indices_train)
        
        logging.info(f"Input shapes - x_train: {x_train.shape}, y_train: {y_train.shape}, point_indices_train: {point_indices_train.shape}")
        
        # Convert to PyTorch tensors with explicit memory layout
        x_train_tensor = torch.from_numpy(x_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        point_indices_train_tensor = torch.from_numpy(point_indices_train).long()
        
        # Ensure tensors are contiguous
        x_train_tensor = x_train_tensor.contiguous()
        y_train_tensor = y_train_tensor.contiguous()
        point_indices_train_tensor = point_indices_train_tensor.contiguous()
        
        # Reshape if needed
        if len(x_train_tensor.shape) == 2:
            x_train_tensor = x_train_tensor.reshape(-1, x_train_tensor.shape[1], 1)
        
        if len(point_indices_train_tensor.shape) == 1:
            point_indices_train_tensor = point_indices_train_tensor.reshape(-1, 1)
        
        # Create training dataset
        self.train_dataset = torch.utils.data.TensorDataset(
            x_train_tensor, point_indices_train_tensor, y_train_tensor
        )
        
        # Create validation dataset if validation data is available
        if self.x_val is not None and self.y_val is not None and self.point_indices_val is not None:
            x_val = np.ascontiguousarray(self.x_val)
            y_val = np.ascontiguousarray(self.y_val)
            point_indices_val = np.ascontiguousarray(self.point_indices_val)
            
            x_val_tensor = torch.from_numpy(x_val).float()
            y_val_tensor = torch.from_numpy(y_val).float()
            point_indices_val_tensor = torch.from_numpy(point_indices_val).long()
            
            # Ensure tensors are contiguous
            x_val_tensor = x_val_tensor.contiguous()
            y_val_tensor = y_val_tensor.contiguous()
            point_indices_val_tensor = point_indices_val_tensor.contiguous()
            
            # Reshape if needed
            if len(x_val_tensor.shape) == 2:
                x_val_tensor = x_val_tensor.reshape(-1, x_val_tensor.shape[1], 1)
            
            if len(point_indices_val_tensor.shape) == 1:
                point_indices_val_tensor = point_indices_val_tensor.reshape(-1, 1)
            
            self.val_dataset = torch.utils.data.TensorDataset(
                x_val_tensor, point_indices_val_tensor, y_val_tensor
            )
            
            # Create prediction dataset (using validation data)
            self.predict_dataset = torch.utils.data.TensorDataset(
                x_val_tensor, point_indices_val_tensor  # Only input features, no targets
            )
        
        # Create test dataset if test data is available
        if self.x_test is not None and self.y_test is not None and self.point_indices_test is not None:
            x_test = np.ascontiguousarray(self.x_test)
            y_test = np.ascontiguousarray(self.y_test)
            point_indices_test = np.ascontiguousarray(self.point_indices_test)
            
            x_test_tensor = torch.from_numpy(x_test).float()
            y_test_tensor = torch.from_numpy(y_test).float()
            point_indices_test_tensor = torch.from_numpy(point_indices_test).long()
            
            # Ensure tensors are contiguous
            x_test_tensor = x_test_tensor.contiguous()
            y_test_tensor = y_test_tensor.contiguous()
            point_indices_test_tensor = point_indices_test_tensor.contiguous()
            
            # Reshape if needed
            if len(x_test_tensor.shape) == 2:
                x_test_tensor = x_test_tensor.reshape(-1, x_test_tensor.shape[1], 1)
            
            if len(point_indices_test_tensor.shape) == 1:
                point_indices_test_tensor = point_indices_test_tensor.reshape(-1, 1)
            
            self.test_dataset = torch.utils.data.TensorDataset(
                x_test_tensor, point_indices_test_tensor, y_test_tensor
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_dataset is None:
            return None
        
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        if self.test_dataset is None:
            # If no test dataset is available, use validation dataset
            if self.val_dataset is None:
                return None
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=2 if self.num_workers > 0 else None
            )
        
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def predict_dataloader(self):
        """Return prediction dataloader."""
        if self.predict_dataset is None:
            # If no prediction dataset is available, use test dataset
            if self.test_dataset is None:
                # If no test dataset, use validation dataset
                if self.val_dataset is None:
                    return None
                # Create prediction dataset from validation data
                self.predict_dataset = torch.utils.data.TensorDataset(
                    self.val_dataset.tensors[0],  # x
                    self.val_dataset.tensors[1]   # point_indices
                )
            else:
                # Create prediction dataset from test data
                self.predict_dataset = torch.utils.data.TensorDataset(
                    self.test_dataset.tensors[0],  # x
                    self.test_dataset.tensors[1]   # point_indices
                )
        
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )


def train_model(
    model: pl.LightningModule,
    data_module: TimeSeriesDataModule,
    max_epochs: int = 100,
    early_stopping_patience: int = 15,
    checkpoint_dir: str = './checkpoints',
    model_name: str = 'timeseries_model',
    gpus: int = 1 if torch.cuda.is_available() else 0
) -> pl.LightningModule:
    """
    Train a PyTorch Lightning model.
    
    Args:
        model: Model to train
        data_module: Data module containing the data
        max_epochs: Maximum number of epochs to train
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        checkpoint_dir: Directory to save checkpoints
        model_name: Name of the model for checkpoint files
        gpus: Number of GPUs to use (defaults to 1 if available, 0 otherwise)
        
    Returns:
        Trained model
    """
    # Set up checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, model_name),
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # Set up early stopping
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        verbose=True,
        mode='min'
    )
    
    # Set up learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else None
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    best_model = model.__class__.load_from_checkpoint(best_model_path)
    
    logging.info(f"Training completed. Best model saved at: {best_model_path}")
    
    return best_model


def evaluate_model(
    model: pl.LightningModule,
    data_module: TimeSeriesDataModule,
    gpus: int = 1 if torch.cuda.is_available() else 0
) -> Dict[str, float]:
    """
    Evaluate a PyTorch Lightning model on test data.
    
    Args:
        model: Model to evaluate
        data_module: Data module containing the data
        gpus: Number of GPUs to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Set up trainer for evaluation
    trainer = pl.Trainer(
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else None
    )
    
    # Evaluate model
    test_results = trainer.test(model, data_module)
    
    return test_results[0] if test_results else {} 