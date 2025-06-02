"""
Main script for multi-point time series forecasting.
"""

import os
import argparse
import logging
import numpy as np
from tensorflow.keras.layers import LeakyReLU, ReLU, PReLU, ELU
import pandas as pd
import matplotlib.pyplot as plt

# Set CUDA environment variables
cuda_path = '/home/yolo/miniconda3/envs/p310-cu121'
nvvm_path = os.path.join(cuda_path, 'nvvm')
cuda_lib_path = os.path.join(cuda_path, 'lib')

# Set CUDA environment variables
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={nvvm_path}'
os.environ['TF_CUDA_PATHS'] = nvvm_path
os.environ['CUDA_PATH'] = cuda_path
os.environ['CUDA_HOME'] = cuda_path
os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Add CUDA library paths to system path
import sys
sys.path.extend([
    cuda_lib_path,
    os.path.join(cuda_path, 'include'),
    os.path.join(nvvm_path, 'lib')
])

# Verify CUDA installation
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("CUDA available:", tf.test.is_built_with_cuda())

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



from src.data_preprocessing import load_data, split_train_test, load_geo_data, DataPreprocessor
from src.trainer import MultiPointTrainer
from src.visualization import plot_loss, plot_predictions, plot_input_sequences, plot_model_comparison
from src.logger import setup_logger
from src.fine_tuning import ModelFineTuner
from src.clustering import KMeansClusterer, DBSCANClusterer

# Disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable pandas warnings
import warnings
warnings.filterwarnings('ignore')

# Disable numpy warnings
np.seterr(all='ignore')


def get_default_param_space():
    """Get default parameter space for Bayesian optimization."""
    return {
        # LSTM parameters
        'lstm_units': ["64_32", "128_64", "64_64_32"],
        'dropout_rates': ["0.2_0.2", "0.3_0.3", "0.2_0.2_0.2"], 
        'dense_units': ["32", "64", "32_16"],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [2048],
        'epochs': [100],
        'point_embedding_dim': [8, 16],
        'use_point_embeddings': [True],
        'use_residual_connections': [True],
        
        # ML parameters
        'ml_learning_rate': [0.01, 0.1, 0.3],
        'ml_max_depth': [3, 5, 7, 10],
        'ml_n_estimators': [50, 100, 200],

        # Convolutional Autoencoder parameters
        'conv_filters': ["32_64_128", "64_128_256", "32_64_128_256"],
        'kernel_sizes': ["3_3_3", "5_5_5", "3_5_7"],
        'encoder_units': ["128_64", "256_128", "512_256"],
        'decoder_units': ["64_128", "128_256", "256_512"],
        'latent_dim': [8, 16, 32],

        # Dense Autoencoder parameters
        'encoder_units': ["256_128_64", "512_256_128", "1024_512_256"],
        'decoder_units': ["64_128_256", "128_256_512", "256_512_1024"]
    }


def create_trainer(model_type: str, train_data: pd.DataFrame, test_data: pd.DataFrame, param_space: dict) -> MultiPointTrainer:
    """
    Create a trainer instance for the specified model type.
    
    Args:
        model_type: Type of model to create trainer for
        train_data: Training data
        test_data: Test data
        param_space: Parameter space for optimization
        
    Returns:
        MultiPointTrainer instance
    """
    # Create model-specific output directory
    model_output_dir = os.path.join("outputs", model_type.upper())
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Create clusterer
    clusterer = DBSCANClusterer(
        min_samples=4,
        n_neighbors=4,
        n_components=2
    )
    
    # Create trainer
    trainer = MultiPointTrainer(
        train_data=train_data,
        test_data=test_data,
        param_space=param_space,
        sequence_length=30,  # Default values, you might want to make these configurable
        output_sequence_length=5,
        output_dir=model_output_dir,
        log_dir="logs",
        n_iter=5,
        model_type=model_type.lower(),
        clusterer=clusterer,
        min_correlation=0.7
    )
    
    return trainer


def main(args):
    """
    Main function to run the time series forecasting pipeline.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    logger = setup_logger(
        name="main",
        log_dir=args.log_dir,
        level=args.log_level
    )
    
    try:
        # Load and split data
        logger.info("Loading data from %s", args.input_file)
        data = load_data(args.input_file)
        data_geo = load_geo_data(args.input_geo_file)

        train_data, test_data = split_train_test(data, args.split_date)
        logger.info(
            "Data split complete - Training size: %d, Test size: %d",
            len(train_data),
            len(test_data)
        )
        

        # Initialize data preprocessor
        data_preprocessor = DataPreprocessor(
            sequence_length=args.sequence_length,
            output_sequence_length=args.output_sequence_length,
            time_features=args.time_features.split(',') if args.time_features else None,
            clusterer=DBSCANClusterer(
                min_samples=args.min_samples,
                n_neighbors=4,
                n_components=2
            ),
            min_correlation=args.min_correlation,
            geo_clustering=args.geo_clustering,
            max_points=args.max_points if hasattr(args, 'max_points') else 1000,
            max_depth=args.max_depth if hasattr(args, 'max_depth') else 5
        )

                # Preprocess data once for all models
        logger.info("Preprocessing data for all models...")
        preprocessed_data = data_preprocessor.preprocess_data(
            train_data=train_data,
            test_data=test_data,
            geo_df=data_geo
        )
        # Get parameter space
        param_space = get_default_param_space()
        
        # Dictionary to store predictions from all models
        all_predictions = {}
        
        # Train and evaluate each model type
        model_types = ['LSTM', 'CONV_AUTOENCODER', 'DENSE_AUTOENCODER', 'ML']
        for model_type in model_types:
            logger.info(f"Training and evaluating {model_type}...")
            
            # Create trainer with proper configuration from args
            trainer = MultiPointTrainer(
                train_data=train_data,
                test_data=test_data,
                param_space=param_space,
                sequence_length=args.sequence_length,
                output_sequence_length=args.output_sequence_length,
                time_features=args.time_features.split(',') if args.time_features else None,
                output_dir=os.path.join(args.output_dir, model_type.upper()),
                log_dir=args.log_dir,
                n_iter=args.n_iter,
                model_type=model_type.lower(),
                clusterer=DBSCANClusterer(
                    min_samples=args.min_samples,
                    n_neighbors=4,
                    n_components=2
                ),
                preprocessor=data_preprocessor.preprocessor
            )
                        # Prepare data for specific model type
            model_data = data_preprocessor.prepare_model_data(
                preprocessed_data,
                model_type.lower()
            )
            # Train and evaluate
            best_params, best_metrics, predictions_df, model = trainer.train_and_evaluate(preprocessed_data=model_data)
            
            # Generate predictions
            logger.info(f"Generating predictions for all points using {model_type}...")
            predictions = trainer.generate_predictions_for_all_test_points(generate_plots=args.generate_plots)
            all_predictions[model_type.lower()] = predictions
            
            # Save predictions
            predictions_file = os.path.join(args.output_dir, model_type.upper(), "predictions.csv")
            predictions.to_csv(predictions_file)
        
        # Create comparison plots
        logger.info("Generating model comparison plots...")
        plot_model_comparison(
            actual_data=test_data,
            predictions_dict=all_predictions,
            point_names=trainer.all_point_columns,
            output_dir=args.output_dir,
            show_confidence=True
        )
        
        logger.info("All models training and comparison completed successfully!")
        
    except Exception as e:
        logger.error("Error in main execution: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Point Time Series Forecasting")
    
    # Existing arguments groups
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        "--input_file",
        type=str,
        default="data/wroclaw_6.csv",
        help="Path to input CSV file"
    )

    data_group.add_argument(
        "--input_geo_file",
        type=str,
        default="data/wroclaw_geo.csv",
        help="Path to input CSV file"
    )

    # Add new clustering arguments group
    clustering_group = parser.add_argument_group('Clustering Parameters')
    clustering_group.add_argument(
        "--clustering_method",
        type=str,
        choices=["kmeans", "dbscan"],
        default="dbscan",
        help="Clustering method to use for data preprocessing"
    )
    clustering_group.add_argument(
        "--n_clusters",
        type=int,
        default=20,
        help="Number of clusters (for KMeans)"
    )
    clustering_group.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for KMeans"
    )

    clustering_group.add_argument(
        "--geo_clustering",
        action="store_true",
        help="Whether to perform geo clustering"
    )

    clustering_group.add_argument(
        "--min_samples",
        type=int,
        default=4,
        help="Minimum samples in neighborhood (for DBSCAN)"
    )
    clustering_group.add_argument(
        "--min_correlation",
        type=float,
        default=0.5,
        help="Minimum correlation threshold for point selection"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Logging level"
    )
    
    parser.add_argument(
        "--split_date",
        type=str,
        default="2020-12-31",
        help="Date to split train/test data"
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=30,
        help="Length of input sequences"
    )
    
    parser.add_argument(
        "--output_sequence_length",
        type=int,
        default=5,
        help="Length of output sequences"
    )
    
    parser.add_argument(
        "--time_features",
        type=str,
        default=None,
        help="Comma-separated list of time features to include (month,day_of_week,day_of_year)"
    )
    
    parser.add_argument(
        "--n_iter",
        type=int,
        default=5,
        help="Number of iterations for Bayesian optimization"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['lstm', 'conv_autoencoder', 'dense_autoencoder', 'ml', 'all'],
        default='all',
        help="Type of model to use. Use 'all' to run all models"
    )
    
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Whether to perform fine-tuning after global model training"
    )
    
    parser.add_argument(
        "--fine_tune_lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning"
    )
    
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=10,
        help="Number of epochs for fine-tuning"
    )
    
    parser.add_argument(
        "--fine_tune_batch_size",
        type=int,
        default=32,
        help="Batch size for fine-tuning"
    )
    
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Whether to generate prediction plots (can be time-consuming)"
    )

    preprocessing_group = parser.add_argument_group('Data Preprocessing Parameters')
    preprocessing_group.add_argument(
        "--max_points",
        type=int,
        default=5000,
        help="Maximum number of points per leaf in quadtree clustering"
    )
    preprocessing_group.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of quadtree clustering"
    )
    
    
    args = parser.parse_args()
    
    # Convert log level string to logging constant
    args.log_level = getattr(logging, args.log_level.upper())
    
    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args) 