"""
Logging module for time series forecasting system.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import sys
import traceback


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_logging: bool = False
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_dir: Directory to save log files
        level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        json_logging: Whether to use JSON structured logging
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    json_formatter = JsonFormatter()
    color_formatter = CustomFormatter()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)
    
    # File handlers
    if log_to_file:
        # Create log directory
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Standard log file
        log_file = log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(standard_formatter)
        logger.addHandler(file_handler)
        
        # JSON log file if requested
        if json_logging:
            json_file = log_dir / f"{name}_{timestamp}_structured.json"
            json_handler = logging.FileHandler(json_file)
            json_handler.setLevel(level)
            json_handler.setFormatter(json_formatter)
            logger.addHandler(json_handler)
    
    # Log initial message
    logger.info(f"Logger {name} initialized with level {logging.getLevelName(level)}")
    if log_to_file:
        logger.info(f"Logging to directory: {log_dir}")
    
    return logger


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        try:
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'name': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            if hasattr(record, 'metrics'):
                log_data['metrics'] = record.metrics
            if hasattr(record, 'params'):
                log_data['parameters'] = record.params
            
            if record.exc_info:
                log_data['exception'] = {
                    'type': str(record.exc_info[0].__name__),
                    'message': str(record.exc_info[1]),
                    'traceback': traceback.format_exception(*record.exc_info)
                }
            
            return json.dumps(log_data)
        except Exception as e:
            return json.dumps({
                'error': 'Error formatting log record',
                'details': str(e)
            })


class TrainingLogger:
    """Handles logging for the training process."""
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        json_logging: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            console_level: Logging level for console output
            file_level: Logging level for file output
            json_logging: Whether to enable JSON structured logging
        """
        self.logger = setup_logger(
            name=name,
            log_dir=log_dir,
            level=min(console_level, file_level),
            log_to_file=True,
            log_to_console=True,
            json_logging=json_logging
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)
    
    def _log(self, level: int, message: str, exc_info: bool = False, **kwargs):
        """Internal logging method."""
        extra = {}
        if kwargs:
            extra.update(kwargs)
        self.logger.log(level, message, exc_info=exc_info, extra=extra)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """Log metrics with optional step and parameters."""
        message = f"Metrics at step {step if step is not None else 'N/A'}"
        self.info(message, metrics=metrics, params=params)
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log model or training parameters."""
        self.info("Parameters", params=params)
    
    def log_model_summary(self, model: 'tf.keras.Model'):
        """Log model architecture summary."""
        # Redirect model summary to string
        from io import StringIO
        summary_io = StringIO()
        model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        self.info(f"Model Summary:\n{summary_io.getvalue()}")
    
    def log_training_start(
        self,
        model_type: str,
        total_params: int,
        trainable_params: int,
        non_trainable_params: int
    ):
        """Log training start with model information."""
        self.info(
            f"Starting training for {model_type} model",
            params={
                'total_params': total_params,
                'trainable_params': trainable_params,
                'non_trainable_params': non_trainable_params
            }
        )
    
    def log_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        learning_rate: float
    ):
        """Log end of epoch with metrics."""
        self.info(
            f"Epoch {epoch} completed",
            metrics=metrics,
            params={'learning_rate': learning_rate}
        )
    
    def log_prediction(
        self,
        point_name: str,
        actual: float,
        predicted: float,
        timestamp: Union[str, datetime]
    ):
        """Log individual prediction."""
        self.debug(
            f"Prediction for {point_name}",
            params={
                'timestamp': str(timestamp),
                'actual': actual,
                'predicted': predicted,
                'error': abs(actual - predicted)
            }
        )
    
    def log_memory_usage(self):
        """Log current memory usage."""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.info(
            "Memory Usage",
            params={
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024
            }
        )
    
    def log_gpu_usage(self):
        """Log GPU memory usage if available."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    memory_info = tf.config.experimental.get_memory_info(gpu.name)
                    self.info(
                        f"GPU {gpu.name} Memory",
                        params={
                            'current_mb': memory_info['current'] / 1024 / 1024,
                            'peak_mb': memory_info['peak'] / 1024 / 1024
                        }
                    )
        except Exception as e:
            self.warning(f"Could not get GPU memory info: {str(e)}")
    
    def log_system_info(self):
        """Log system information."""
        import platform
        system_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count()
        }
        self.info("System Info", params=system_info)
    
    def create_experiment_log(
        self,
        experiment_name: str,
        config: Dict[str, Any]
    ) -> Path:
        """Create new experiment log directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path(self.logger.handlers[0].baseFilename).parent / f"{experiment_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_file = experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.info(f"Created experiment directory: {experiment_dir}")
        return experiment_dir 