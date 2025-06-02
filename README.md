# Multi-Point Time Series Forecasting

This project implements a comprehensive time series forecasting system that supports multiple model architectures and can handle multiple time series points simultaneously.

## Features

- Multiple model architectures:
  - LSTM (Long Short-Term Memory)
  - Convolutional Autoencoder
  - Dense Autoencoder
  - ML (Traditional Machine Learning)
- Bayesian hyperparameter optimization
- Multi-point time series forecasting
- Fine-tuning capabilities
- Comprehensive visualization tools
- Detailed performance metrics and comparisons

## Requirements

- Python 3.8+
- Conda package manager
- CUDA-compatible GPU (optional but recommended)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate Conda environment:
```bash
conda env create -f environment.yaml
conda activate time_series_env
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Format

The input data should be a CSV file with the following format:
- First column: Date (in YYYY-MM-DD format)
- Subsequent columns: Time series values for different points
- Missing values should be marked as NaN

Example:
```csv
Date,Point1,Point2,Point3
2020-01-01,23.5,45.2,67.8
2020-01-02,24.1,46.0,68.2
...
```

## Usage

### Basic Usage

Run all models:
```bash
python main.py --input_file data/your_data.csv
```

Run a specific model:
```bash
python main.py --input_file data/your_data.csv --model_type lstm
```

### Command Line Arguments

- `--input_file`: Path to input CSV file (default: "data/wroclaw.csv")
- `--output_dir`: Directory to save outputs (default: "outputs")
- `--log_dir`: Directory to save logs (default: "logs")
- `--model_type`: Model type to use ['lstm', 'conv_autoencoder', 'dense_autoencoder', 'ml', 'all'] (default: 'all')
- `--clustering_method`: Clustering method to use ['kmeans', 'dbscan']
- `--geo_clustering`: Whether to perform geo clustering with QuadTree Clustering, if yes the file with coordinates is needed
- `--split_date`: Date to split train/test data (default: "2020-12-31")
- `--sequence_length`: Length of input sequences (default: 30)
- `--output_sequence_length`: Length of output sequences (default: 5)
- `--time_features`: Comma-separated list of time features (default: None)
- `--n_iter`: Number of iterations for Bayesian optimization (default: 5)
- `--fine_tune`: Enable fine-tuning (flag)
- `--fine_tune_lr`: Learning rate for fine-tuning (default: 1e-4)
- `--fine_tune_epochs`: Number of epochs for fine-tuning (default: 10)
- `--fine_tune_batch_size`: Batch size for fine-tuning (default: 32)
- `--generate_plots`: Generate prediction plots (flag)
- `--log_level`: Logging level ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] (default: 'INFO')

## How It Works

### 1. Data Processing
- Loads time series data from CSV file
- Splits data into training and test sets based on specified date
- Preprocesses data including normalization and feature engineering

### 2. Model Architecture

#### LSTM Model
- Multi-layer LSTM architecture
- Point embeddings for handling multiple time series
- Residual connections for better gradient flow
- Configurable dropout rates and unit sizes

#### Convolutional Autoencoder
- Encoder: Conv1D layers for feature extraction
- Decoder: Transposed Conv1D layers for reconstruction
- Bottleneck layer for compressed representation
- Skip connections for preserving information

#### Dense Autoencoder
- Fully connected layers for encoding and decoding
- Symmetric architecture
- Configurable layer sizes and activation functions

#### ML Model
- Traditional machine learning approaches
- Support for various algorithms (Random Forest, XGBoost, etc.)
- Feature engineering for time series data

### 3. Training Process
1. Performs Bayesian optimization for hyperparameter tuning
2. Trains models with optimal parameters
3. Optional fine-tuning for individual points
4. Generates predictions for test set

### 4. Output and Evaluation
- Saves trained models and parameters
- Generates prediction plots (if enabled)
- Calculates performance metrics (RMSE, MAE, MAPE)
- Creates comparative analysis between models
- Detailed logging of training process

## Output Structure

```
outputs/
├── LSTM/
│   ├── best_parameters.txt
│   ├── all_points_predictions.csv
│   └── plots/
├── CONV_AUTOENCODER/
│   ├── ...
├── DENSE_AUTOENCODER/
│   ├── ...
├── ML/
│   ├── ...
└── model_comparison/
    ├── metrics_comparison.csv
    └── predictions_comparison_*.png
```

## Logging

The script provides detailed logging at different levels:
- DEBUG: Detailed information for debugging
- INFO: General information about progress
- WARNING: Warning messages for potential issues
- ERROR: Error messages for serious problems
- CRITICAL: Critical errors that stop execution

Logs are saved in the specified log directory with timestamps.

## Performance Metrics

For each model and point, the following metrics are calculated:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

Results are saved in CSV format for easy comparison and analysis.
