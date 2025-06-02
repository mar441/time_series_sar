"""
Data preprocessing module for multi-point time series forecasting.
Handles data loading, cleaning, and sequence creation for panel time series data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from typing import Tuple, Dict, List, Union, Optional, Any
import os
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from .clustering import TimeSeriesClusterer, KMeansClusterer, DBSCANClusterer
from src.logger import TrainingLogger
from src.quadtree_clustering import QuadTreeClusterer, quadtree_clustering

def handle_missing_values(
    data: pd.DataFrame,
    method: str = 'interpolate',
    max_gap: int = 5,
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Handle missing values in the dataset using various methods.
    
    Args:
        data: Input DataFrame
        method: Method to handle missing values ('interpolate', 'ffill', 'mean')
        max_gap: Maximum gap to interpolate
        min_periods: Minimum periods for rolling statistics
        
    Returns:
        DataFrame with handled missing values
    """
    # Make a copy to avoid modifying original data
    data_cleaned = data.copy()
    
    # Get numeric columns
    numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        # Check for missing values
        if data_cleaned[column].isnull().any():
            if method == 'interpolate':
                # Linear interpolation for small gaps
                data_cleaned[column] = data_cleaned[column].interpolate(
                    method='linear',
                    limit=max_gap,
                    limit_direction='both'
                )
                
                # Fill remaining gaps with rolling mean
                rolling_mean = data_cleaned[column].rolling(
                    window=min_periods*2,
                    min_periods=min_periods,
                    center=True
                ).mean()
                
                data_cleaned[column].fillna(rolling_mean, inplace=True)
                
            elif method == 'ffill':
                # Forward fill with limit
                data_cleaned[column].fillna(
                    method='ffill',
                    limit=max_gap,
                    inplace=True
                )
                # Backward fill remaining
                data_cleaned[column].fillna(
                    method='bfill',
                    limit=max_gap,
                    inplace=True
                )
                
            elif method == 'mean':
                # Fill with rolling mean
                rolling_mean = data_cleaned[column].rolling(
                    window=min_periods*2,
                    min_periods=min_periods,
                    center=True
                ).mean()
                data_cleaned[column].fillna(rolling_mean, inplace=True)
                
            # Fill any remaining missing values with column mean
            if data_cleaned[column].isnull().any():
                data_cleaned[column].fillna(data_cleaned[column].mean(), inplace=True)
    
    return data_cleaned


@tf.function
def create_sequences_optimized(
    data: tf.Tensor,
    sequence_length: int,
    output_sequence_length: int,
    step: int = 1
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Create sequences using TensorFlow operations for better performance.
    
    Args:
        data: Input tensor of shape [time_steps, features]
        sequence_length: Length of input sequences
        output_sequence_length: Length of target sequences
        step: Step size between sequences
        
    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    # Calculate dimensions
    total_length = sequence_length + output_sequence_length
    n_sequences = (tf.shape(data)[0] - total_length) // step + 1
    
    # Create sequence indices
    indices = tf.range(0, n_sequences * step, step)
    
    # Create input sequences
    input_sequences = tf.map_fn(
        lambda x: data[x:x + sequence_length],
        indices,
        dtype=tf.float32
    )
    
    # Create target sequences
    target_sequences = tf.map_fn(
        lambda x: data[x + sequence_length:x + total_length],
        indices,
        dtype=tf.float32
    )
    
    return input_sequences, target_sequences


def create_panel_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    output_sequence_length: int,
    step: int = 6,
    time_features: List[str] = None,
    batch_size: int = 1024,
    use_tf_optimization: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create input/output sequences for multi-point time series prediction with batching.
    
    Args:
        data: Panel DataFrame with 'Date' and multiple point columns
        sequence_length: Length of input sequences
        output_sequence_length: Length of output sequences
        step: Step size between sequences
        time_features: List of time-based features to include
        batch_size: Size of batches for processing
        use_tf_optimization: Whether to use TF-optimized sequence creation
        
    Returns:
        Tuple of (input_sequences, target_sequences, point_indices)
    """
    point_columns = [col for col in data.columns if col not in ['Date'] + (time_features or [])]
    num_points = len(point_columns)
    
    sequences = []
    targets = []
    point_indices = []
    
    # Process points in batches
    for batch_start in tqdm(range(0, num_points, batch_size), desc="Creating sequences"):
        batch_end = min(batch_start + batch_size, num_points)
        batch_columns = point_columns[batch_start:batch_end]
        
        for point_idx, column in enumerate(batch_columns, start=batch_start):
            point_data = data[column].values
            
            if use_tf_optimization:
                # Convert to TensorFlow tensor
                tf_data = tf.convert_to_tensor(point_data, dtype=tf.float32)
                seq_batch, target_batch = create_sequences_optimized(
                    tf_data,
                    sequence_length,
                    output_sequence_length,
                    step
                )
                
                # Convert back to numpy
                seq_batch = seq_batch.numpy()
                target_batch = target_batch.numpy()
                
            else:
                seq_batch = []
                target_batch = []
                for i in range(0, len(point_data) - sequence_length - output_sequence_length + 1, step):
                    seq = point_data[i:i + sequence_length]
                    target = point_data[i + sequence_length:i + sequence_length + output_sequence_length]
                    seq_batch.append(seq)
                    target_batch.append(target)
                
                seq_batch = np.array(seq_batch)
                target_batch = np.array(target_batch)
            
            # Add time features if specified
            if time_features:
                time_data = data[time_features].iloc[:-output_sequence_length].values
                time_sequences = []
                for i in range(0, len(time_data) - sequence_length + 1, step):
                    time_seq = time_data[i:i + sequence_length]
                    time_sequences.append(time_seq)
                time_sequences = np.array(time_sequences)
                seq_batch = np.concatenate([seq_batch.reshape(-1, sequence_length, 1), time_sequences], axis=2)
            else:
                seq_batch = seq_batch.reshape(-1, sequence_length, 1)
            
            sequences.append(seq_batch)
            targets.append(target_batch)
            point_indices.extend([point_idx] * len(seq_batch))
    
    # Concatenate all batches
    sequences = np.concatenate(sequences, axis=0)
    targets = np.concatenate(targets, axis=0)
    point_indices = np.array(point_indices)
    
    return sequences, targets, point_indices


def select_representative_points(
    data: pd.DataFrame,
    clusterer: TimeSeriesClusterer,
    min_correlation: float = 0.5,
    time_column: str = 'Date'
) -> List[str]:
    """
    Select representative points using clustering and correlation analysis.
    
    Args:
        data: DataFrame with time series data
        clusterer: Instance of TimeSeriesClusterer to use for clustering
        min_correlation: Minimum correlation threshold
        time_column: Name of the time column
        
    Returns:
        List of selected column names
        
    Example:
    ```python
    # Using KMeans clustering
    kmeans_clusterer = KMeansClusterer(n_clusters=10, random_state=42)
    selected_points = select_representative_points(data, kmeans_clusterer)
    
    # Using DBSCAN clustering
    dbscan_clusterer = DBSCANClusterer(eps=0.5, min_samples=5)
    selected_points = select_representative_points(data, dbscan_clusterer)
    ```
    """
    if time_column in data.columns:
        data_no_time = data.drop(columns=[time_column])
    else:
        data_no_time = data.copy() 
        
    point_columns = data_no_time.columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = data_no_time.corr()
    
    # Prepare data for clustering
    data_array = data_no_time.values.T  # Transpose to get points as rows
    
    # Perform clustering
    cluster_labels = clusterer.fit(data_array)
    
    # Select representative points from each cluster
    selected_points = []
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        # Skip noise points (labeled as -1) if using DBSCAN
        if cluster_id == -1:
            continue
            
        # Get points in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_points = [point_columns[i] for i in cluster_indices]
        
        if len(cluster_points) == 1:
            selected_points.append(cluster_points[0])
            continue
        
        # Find the most representative point (highest average correlation with others)
        max_avg_corr = -1
        representative = None
        
        for point in cluster_points:
            # Calculate average correlation with other points in cluster
            correlations = [
                abs(corr_matrix.loc[point, other])
                for other in cluster_points
                if other != point
            ]
            avg_corr = np.mean(correlations)
            
            if avg_corr > max_avg_corr:
                max_avg_corr = avg_corr
                representative = point
        
        if representative and max_avg_corr >= min_correlation:
            selected_points.append(representative)

    print(f'Number of selected points: {len(selected_points)}')   
    return selected_points



def load_geo_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with validation and error handling.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with loaded data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load data
        data = pd.read_csv(file_path)

        # Check for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found in data")
        return data

    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise



def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with validation and error handling.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data
        data = pd.read_csv(file_path)
        
        # Validate Date column
        if 'Date' not in data.columns:
            raise ValueError("Data must contain 'Date' column")
        
        # Convert Date to datetime and validate
        try:
            data['Date'] = pd.to_datetime(data['Date'])
        except Exception as e:
            raise ValueError(f"Error converting Date column to datetime: {str(e)}")
        
        # Check for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found in data")
        
        # Set Date as index
        data.set_index('Date', inplace=True)
        
        # Handle missing values
        data = handle_missing_values(data)
        
        return data
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise


def split_train_test(
    data: pd.DataFrame,
    split_date: str,
    min_train_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets based on date with validation.
    
    Args:
        data: Input DataFrame
        split_date: Date to split on (YYYY-MM-DD)
        min_train_size: Minimum required size of training set as fraction
        
    Returns:
        Tuple of (train_data, test_data)
    """
    try:
        # Convert split_date to datetime
        split_date = pd.to_datetime(split_date)
        
        # Validate split date
        if split_date <= data.index.min() or split_date >= data.index.max():
            raise ValueError("Split date must be within data range")
        
        # Split data
        train_data = data[data.index <= split_date].copy()
        test_data = data[data.index > split_date].copy()
        
        # Validate split sizes
        total_size = len(data)
        train_size = len(train_data)
        train_fraction = train_size / total_size
        
        if train_fraction < min_train_size:
            raise ValueError(f"Training set too small: {train_fraction:.2%} < {min_train_size:.2%}")
        
        print("\nSplit data info:")
        print(f"Train shape: {train_data.shape}")
        print(f"Test shape: {test_data.shape}")
        print(f"Train period: {train_data.index.min()} to {train_data.index.max()}")
        print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")
        
        return train_data, test_data
        
    except Exception as e:
        logging.error(f"Error splitting data: {str(e)}")
        raise



class DataPreprocessor:
    """Handles main data preprocessing for all models."""
    
    def __init__(
        self,
        sequence_length: int,
        output_sequence_length: int,
        normalization_range: Tuple[int, int] = (-1, 1),
        time_features: Optional[List[str]] = None,
        clusterer: Optional[TimeSeriesClusterer] = None,
        min_correlation: float = 0.5,
        geo_clustering: bool = False,
        max_points: int = 10,
        max_depth: int = 5
    ):
        """
        Initialize data preprocessor.
        
        Args:
            sequence_length: Length of input sequences
            output_sequence_length: Length of output sequences
            normalization_range: Range for data normalization
            time_features: List of time features to include
            clusterer: TimeSeriesClusterer instance to use
            min_correlation: Minimum correlation threshold
            geo_clustering: Whether to use geographical clustering
            max_points: Maximum number of points per leaf in quadtree clustering
            max_depth: Maximum depth of quadtree clustering
        """
        self.sequence_length = sequence_length
        self.output_sequence_length = output_sequence_length
        self.normalization_range = normalization_range
        self.time_features = time_features
        self.clusterer = clusterer
        self.min_correlation = min_correlation
        self.geo_clustering = geo_clustering
        self.max_points = max_points
        self.max_depth = max_depth
        
        # Initialize preprocessor
        self.preprocessor = MultiPointPreprocessor(
            sequence_length=sequence_length,
            output_sequence_length=output_sequence_length,
            normalization_range=normalization_range,
            time_features=time_features,
        )
        
        # Initialize logger
        self.logger = TrainingLogger(
            name="data_preprocessor",
            log_dir="logs"
        )
    
    def preprocess_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        geo_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Preprocess data for all models.
        
        Args:
            train_data: Training data
            test_data: Test data
            geo_df: Optional geographical data for clustering
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        try:
            # Store all original columns
            all_data = pd.concat([train_data, test_data])
            self.all_point_columns = [col for col in all_data.columns if col != 'Date']
            
            # Select representative points
            selected_points = self._select_representative_points(all_data, geo_df)
            self.preprocessor.point_columns = selected_points
            
            
            # Preprocess training data
            train_sequences, train_targets, train_indices = self.preprocessor.transform(train_data)
            
            # Preprocess test data with historical context
            last_train_data = train_data.iloc[-self.sequence_length:]
            full_test_data = pd.concat([last_train_data, test_data])
            test_sequences, test_targets, test_indices = self.preprocessor.transform(full_test_data)
            
            # Store metadata
            metadata = {
                'num_points': len(selected_points),
                'num_features': 1,
                'sequence_length': self.sequence_length,
                'output_sequence_length': self.output_sequence_length,
                'all_point_columns': self.all_point_columns,
                'selected_points': selected_points
            }
            
            # Store preprocessed data
            preprocessed_data = {
                'train': {
                    'sequences': train_sequences,
                    'targets': train_targets,
                    'indices': train_indices
                },
                'test': {
                    'sequences': test_sequences,
                    'targets': test_targets,
                    'indices': test_indices
                }
            }
            
            return {
                'metadata': metadata,
                'data': preprocessed_data
            }
            
        except Exception as e:
            self.logger.error(f"Error in preprocess_data: {str(e)}", exc_info=True)
            raise
    
    def _select_representative_points(
        self,
        all_data: pd.DataFrame,
        geo_df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """Select representative points using clustering."""
        if self.geo_clustering and geo_df is not None:
            return self._select_points_geo_clustering(all_data, geo_df)
        

        elif self.clusterer is not None:
            return self.preprocessor.select_representative_points(
                all_data,
                clusterer=self.clusterer,
                min_correlation=self.min_correlation
            )
        else:
            return self.preprocessor.select_representative_points(all_data)
    
    def _select_points_geo_clustering(
        self,
        all_data: pd.DataFrame,
        geo_df: pd.DataFrame
    ) -> List[str]:
        """Select points using geographical clustering."""
        try:
            # Filter geo_df to include only points present in all_data
            valid_points = [col for col in all_data.columns if col != 'Date']
            geo_df_filtered = geo_df[geo_df['pid'].isin(valid_points)]
            
            if geo_df_filtered.empty:
                self.logger.warning("No matching points between geo_df and all_data! Using default clustering.")
                return self.preprocessor.select_representative_points(all_data)
            
            # Perform quadtree clustering
            self.logger.info(f"Starting quadtree clustering with max_points={self.max_points}, max_depth={self.max_depth}...")
            geo_df_clustered, quad_stats = quadtree_clustering(
                geo_df=geo_df_filtered,
                max_points_per_leaf=self.max_points,
                min_points_per_leaf=10,
                max_depth=self.max_depth,
                plot=True,
                output_dir="outputs"
            )
            self.logger.info(f"Quadtree clustering completed. Found {len(quad_stats)} clusters.")
            
            # Select representative points from each cluster
            selected_points = []
            valid_clusters = [c for c in geo_df_clustered['quad_id'].unique() if c != -1]
            
            for cluster_id, group in tqdm(geo_df_clustered.groupby('quad_id'),
                                        desc="Processing clusters",
                                        total=len(valid_clusters)):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                # Get points in this cluster
                cluster_points = group['pid'].tolist()
                
                if len(cluster_points) == 1:
                    selected_points.append(cluster_points[0])
                    continue
                
                # Select representative point from cluster
                cluster_data = all_data[cluster_points]
                representative_point = self.preprocessor.select_representative_points(
                    cluster_data,
                    clusterer=self.clusterer,
                    min_correlation=self.min_correlation
                )
                print("len(representative_point)", len(representative_point))
                if representative_point:
                    selected_points.extend(representative_point)
            
            self.logger.info(f"Selected {len(selected_points)} representative points using geospatial clustering")
            return selected_points
            
        except Exception as e:
            self.logger.error(f"Error in geo clustering: {str(e)}", exc_info=True)
            self.logger.warning("Falling back to default clustering")
            return self.preprocessor.select_representative_points(all_data)
    
    def prepare_model_data(
        self,
        preprocessed_data: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Prepare data for specific model type.
        
        Args:
            preprocessed_data: Preprocessed data from preprocess_data
            model_type: Type of model to prepare data for
            
        Returns:
            Dictionary containing model-specific data
        """
        try:
            data = preprocessed_data['data']
            metadata = preprocessed_data['metadata']


            if model_type == 'conv_autoencoder':
                return self._prepare_conv_autoencoder_data(data, metadata)
            elif model_type == 'dense_autoencoder':
                return self._prepare_dense_autoencoder_data(data, metadata)
            else:  # lstm or ml
                return self._prepare_lstm_data(data, metadata)
                
        except Exception as e:
            self.logger.error(f"Error in prepare_model_data: {str(e)}", exc_info=True)
            raise
    
    def _prepare_conv_autoencoder_data(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for convolutional autoencoder."""
        # Calculate required padding for CNN

        num_conv_layers = 3  # Fixed based on parameter space
        pool_size = 2
        total_pooling = pool_size ** num_conv_layers
        padded_sequence_length = ((self.sequence_length + total_pooling - 1) // total_pooling) * total_pooling
        
        # Reshape and pad sequences
        train_sequences = data['train']['sequences']
        test_sequences = data['test']['sequences']
        num_points = metadata['num_points']
        num_features = metadata['num_features']
        
        if len(train_sequences.shape) == 2:
            train_sequences = train_sequences.reshape(-1, self.sequence_length, 1)
            test_sequences = test_sequences.reshape(-1, self.sequence_length, 1)
        
        # Pad sequences
        train_sequences = np.pad(
            train_sequences,
            ((0, 0), (0, padded_sequence_length - self.sequence_length), (0, 0)),
            mode='edge'
        )
        test_sequences = np.pad(
            test_sequences,
            ((0, 0), (0, padded_sequence_length - self.sequence_length), (0, 0)),
            mode='edge'
        )
        
        return {
            'x_train': train_sequences,
            'x_val': test_sequences,
            'y_train': data['train']['targets'],
            'y_val': data['test']['targets'],
            'indices_train': data['train']['indices'],
            'indices_val': data['test']['indices'],
            'num_points': num_points,
            'num_features': num_features,
            'all_point_columns': metadata['all_point_columns']
        }
    
    def _prepare_dense_autoencoder_data(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for dense autoencoder."""
        # Reshape sequences for dense autoencoder
        train_sequences = data['train']['sequences'].reshape(-1, self.sequence_length, 1)
        test_sequences = data['test']['sequences'].reshape(-1, self.sequence_length, 1)
        num_points = metadata['num_points']
        num_features = metadata['num_features']
        return {
            'x_train': train_sequences,
            'x_val': test_sequences,
            'y_train': data['train']['targets'],
            'y_val': data['test']['targets'],
            'indices_train': data['train']['indices'],
            'indices_val': data['test']['indices'],
            'num_points': num_points,
            'num_features': num_features,
            'all_point_columns': metadata['all_point_columns']
        }
    
    def _prepare_lstm_data(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for LSTM model."""
        # Reshape sequences for LSTM
        train_sequences = data['train']['sequences'].reshape(-1, self.sequence_length, 1)
        test_sequences = data['test']['sequences'].reshape(-1, self.sequence_length, 1)
        num_points = metadata['num_points']
        num_features = metadata['num_features']
        # Ensure indices have correct shape
        train_indices = data['train']['indices'].reshape(-1, 1)
        test_indices = data['test']['indices'].reshape(-1, 1)
        
        # Return data in format expected by trainer
        return {
            'x_train': train_sequences,
            'x_val': test_sequences,
            'y_train': data['train']['targets'],
            'y_val': data['test']['targets'],
            'indices_train': train_indices,
            'indices_val': test_indices,    
            'num_points': num_points,
            'num_features': num_features,
            'all_point_columns': metadata['all_point_columns']
        }



class MultiPointPreprocessor:
    """Handles preprocessing of multi-point time series data."""
    
    def __init__(
        self,
        sequence_length: int,
        output_sequence_length: int,
        normalization_range: Tuple[int, int] = (-1, 1),
        time_features: Optional[List[str]] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Length of input sequences
            output_sequence_length: Length of output sequences
            normalization_range: Range for data normalization
            time_features: List of time features to include
        """
        self.sequence_length = sequence_length
        self.output_sequence_length = output_sequence_length
        self.normalization_range = normalization_range
        self.time_features = time_features or []
        self.point_columns = None
        
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Args:
            data: Input DataFrame
        """
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")
            
        # Check for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("Data must contain numeric columns")
            
        # Check for infinite values
        if np.any(np.isinf(data[numeric_cols].values)):
            raise ValueError("Data contains infinite values")
            
        # Check for very large values
        max_allowed = np.finfo(np.float64).max / 1e6  # Leave room for calculations
        if np.any(np.abs(data[numeric_cols].values) > max_allowed):
            raise ValueError(f"Data contains values larger than {max_allowed}")
    
    def _handle_invalid_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle invalid values in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)

        # Clip very large values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        max_allowed = np.finfo(np.float64).max / 1e6
        data[numeric_cols] = data[numeric_cols].clip(-max_allowed, max_allowed)
        
        return data
    
    def select_representative_points(
        self,
        data: pd.DataFrame,
        clusterer: Optional[TimeSeriesClusterer] = None,
        min_correlation: float = 0.5
    ) -> List[str]:
        """
        Select representative points using clustering.
        
        Args:
            data: DataFrame with time series data
            clusterer: TimeSeriesClusterer instance to use (defaults to KMeansClusterer with 20 clusters)
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of selected column names
        """
        if clusterer is None:
            clusterer = KMeansClusterer(n_clusters=20)
            
        return select_representative_points(
            data,
            clusterer=clusterer,
            min_correlation=min_correlation
        )
    
    def _create_sequences(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (sequences, targets, point_indices)
        """
        values = data[self.point_columns].values
        total_sequences = len(values) - self.sequence_length - self.output_sequence_length + 1
        num_points = len(self.point_columns)
        
        # Pre-allocate arrays for sequences
        sequences = []
        targets = []
        point_indices = []
        
        # Create sequences with a step size
        step_size = 1  # Use 6 days as step size
        
        for point_idx in range(num_points):
            point_values = values[:, point_idx]
            # Create sequences for this point
            for i in range(0, total_sequences, step_size):
                # Extract sequence and target
                sequence = point_values[i:i + self.sequence_length]
                target = point_values[i + self.sequence_length:i + self.sequence_length + self.output_sequence_length]
                
                # Only add if we have complete sequence and target
                if len(sequence) == self.sequence_length and len(target) == self.output_sequence_length:
                    sequences.append(sequence)
                    targets.append(target)
                    point_indices.append(point_idx)

        # Convert to numpy arrays
        sequences = np.array(sequences)
        targets = np.array(targets)
        point_indices = np.array(point_indices)
        return sequences, targets, point_indices.reshape(-1, 1)
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        clusterer: Optional[TimeSeriesClusterer] = None,
        min_correlation: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            data: DataFrame with time series data
            clusterer: TimeSeriesClusterer instance to use (defaults to KMeansClusterer)
            min_correlation: Minimum correlation threshold
            
        Returns:
            Tuple of (input_sequences, target_sequences, point_indices)
        """
        self._validate_data(data)
        data = self._handle_invalid_values(data)
        
        # Select representative points
        self.selected_points = self.select_representative_points(
            data,
            clusterer=clusterer,
            min_correlation=min_correlation
        )
        
        return self.transform(data)
    
    def transform(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (sequences, targets, point_indices)
        """
        # Validate and clean data
        data = data[self.point_columns]

        self._validate_data(data)
        data = self._handle_invalid_values(data.copy())
        # Create sequences
        sequences, targets, point_indices = self._create_sequences(data)
        
        # # # Plot comparison for test data
        # plot_data_comparison(
        #     raw_data=data,
        #     sequences=sequences,
        #     targets=targets,
        #     point_indices=point_indices,
        #     point_columns=self.point_columns,
        #     output_dir="./outputs/data_visualization_test"
        # )
        
        return sequences, targets, point_indices
    
    def inverse_transform_values(
        self,
        values: np.ndarray,
        point_indices: np.ndarray
    ) -> np.ndarray:
        """
        Since we're not scaling values, just return them as is.
        
        Args:
            values: Values to inverse transform
            point_indices: Point indices for each value
            
        Returns:
            Same values
        """
        return values
