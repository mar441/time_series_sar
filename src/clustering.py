"""
Clustering module providing various clustering algorithms for time series data.
This module implements a flexible interface for different clustering methods.

How to use different clustering algorithms:
-----------------------------------------
1. In data_preprocessing.py, when calling select_representative_points:
   ```python
   from src.clustering import KMeansClusterer, DBSCANClusterer
   
   # Option 1: Using KMeans
   kmeans = KMeansClusterer(n_clusters=10, random_state=42)
   points = preprocessor.select_representative_points(
       data=your_data,
       clusterer=kmeans,
       min_correlation=0.7
   )
   
   # Option 2: Using DBSCAN
   dbscan = DBSCANClusterer(eps=0.5, min_samples=5)
   points = preprocessor.select_representative_points(
       data=your_data,
       clusterer=dbscan,
       min_correlation=0.7
   )
   
   # Option 3: Using default (KMeans with 20 clusters)
   points = preprocessor.select_representative_points(
       data=your_data,
       min_correlation=0.7
   )
   ```

2. In MultiPointPreprocessor class initialization:
   ```python
   preprocessor = MultiPointPreprocessor(
       sequence_length=24,
       output_sequence_length=12
   )
   
   # Then during fit_transform:
   sequences, targets, indices = preprocessor.fit_transform(
       data=your_data,
       clusterer=KMeansClusterer(n_clusters=15),  # Choose your clusterer here
       min_correlation=0.7
   )
   ```

3. Direct usage of clustering module:
   ```python
   # Create your clusterer
   clusterer = KMeansClusterer(n_clusters=5)
   # or
   clusterer = DBSCANClusterer(eps=0.3, min_samples=3)
   
   # Prepare your data (n_samples x n_features)
   data = np.array([...])
   
   # Get cluster labels
   labels = clusterer.fit(data)
   ```

Potential improvements and extensions:
---------------------------------------
1. Performance Optimizations:
   - Add parallel processing for large datasets using multiprocessing
   - Implement batch processing for memory efficiency
   - Add GPU support for clustering algorithms that support it
   - Cache intermediate results for repeated operations

2. Additional Clustering Methods to Implement:
   - Hierarchical clustering (Agglomerative)
   - Spectral clustering for complex-shaped clusters
   - OPTICS as an improvement over DBSCAN
   - Time series specific clustering methods (e.g., DTW-based)
   - Fuzzy clustering (e.g., Fuzzy C-means)

3. Evaluation and Validation:
   - Add methods to evaluate clustering quality (silhouette score, Davies-Bouldin index)
   - Cross-validation for parameter tuning
   - Methods to visualize clustering results
   - Automatic parameter selection (e.g., optimal eps for DBSCAN)

4. Feature Engineering:
   - Add time series feature extraction (e.g., trend, seasonality, entropy)
   - Support for different distance metrics
   - Dimensionality reduction options (PCA, t-SNE)
   - Add support for handling seasonal patterns

5. Robustness and Error Handling:
   - Add input validation for different data types
   - Handle edge cases (empty clusters, singular matrices)
   - Add support for online/incremental clustering
   - Better handling of outliers

6. Integration Features:
   - Export results to different formats
   - Integration with visualization libraries
   - Save/load trained models
   - REST API interface for remote clustering

7. Documentation and Testing:
   - Add more usage examples
   - Property-based testing
   - Performance benchmarks
   - Interactive tutorials/notebooks

8. Advanced Features:
   - Ensemble clustering methods
   - Semi-supervised clustering options
   - Active learning integration
   - Streaming data support
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import logging
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class TimeSeriesClusterer(ABC):
    """
    Abstract base class for time series clustering algorithms.
    Inherit from this class to implement new clustering methods.
    """
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the clustering algorithm to the data.
        
        Args:
            data: Array of shape (n_samples, n_features) containing the time series data
                 Each row represents a time series point
                 
        Returns:
            np.ndarray: Cluster labels for each point
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the clustering algorithm.
        
        Returns:
            Dict containing the algorithm parameters
        """
        pass


class KMeansClusterer(TimeSeriesClusterer):
    """
    K-means clustering implementation for time series data.
    
    Example usage:
    ```python
    # Initialize clusterer
    kmeans_clusterer = KMeansClusterer(n_clusters=5, random_state=42)
    
    # Prepare your data (n_samples x n_features)
    data = np.array([...])  # Your time series data
    
    # Fit and get cluster labels
    labels = kmeans_clusterer.fit(data)
    
    # Access cluster centers if needed
    centers = kmeans_clusterer.cluster_centers_
    ```
    """
    
    def __init__(self, n_clusters: int = 5, random_state: Optional[int] = None):
        """
        Initialize KMeans clusterer.
        
        Args:
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit KMeans clustering to the data.
        
        Args:
            data: Array of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Cluster labels
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Fit KMeans
        labels = self.model.fit_predict(scaled_data)
        
        # Store cluster centers
        self.cluster_centers_ = self.model.cluster_centers_
        
        print(f'K-means: number of clusters = {n_clusters}')
        
        # Plot clusters in 2D using PCA
        try:
            os.makedirs("outputs/clusters", exist_ok=True)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(scaled_data)
            fig, ax = plt.subplots(figsize=(15, 12))
            unique_labels = np.unique(labels)
            cmap = plt.cm.get_cmap('viridis', len(unique_labels))
            for cluster_id in unique_labels:
                cluster_points = X_pca[labels == cluster_id]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           color=(cmap(cluster_id) if len(unique_labels) > 1 else 'blue'), alpha=0.6)
            ax.set_title(f'Klasteryzacja KMeans (liczba klastrów: {self.model.n_clusters})')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            fig.savefig("outputs/clusters/kmeans_clusters.png", dpi=300)
            plt.close(fig)
        except Exception as e:
            logger.error(f'Failed to plot KMeans clusters: {e}')
        
        return labels
        
    def get_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }


class DBSCANClusterer(TimeSeriesClusterer):
    """
    Implementation of DBSCAN with automatic eps determination using the "knee" method
    (kneed) and mandatory t-SNE(2 components) for dimensionality reduction.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_neighbors: int = 4,
        min_samples: int = 4,
        n_components: int = 2,
        S: float = 1.0,
        curve: str = "convex",
        perplexity: Optional[int] = None
    ):
        """
        Initialize DBSCANClusterer with mandatory t-SNE for dimensionality reduction
        and automatic eps detection using the "knee" method.

        Args:
            n_neighbors (int): Number of neighbors used in NearestNeighbors (to compute eps).
            min_samples (int): Minimum number of samples in a neighborhood to consider 
                               a point a core point in DBSCAN.
            n_components (int): Number of t-SNE components to reduce the data to.
            S (float): Sensitivity parameter used by the kneed library.
            curve (str): Shape of the curve in kneed (e.g., "convex", "concave").
            perplexity (int, optional): Perplexity parameter for t-SNE. If None, will be set to min(30, n_samples-1).
        """
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples
        self.n_components = n_components
        self.S = S
        self.curve = curve
        self.random_state = random_state
        self.perplexity = perplexity

        # The DBSCAN model will be created in fit, after eps is determined
        self.model = None
        
        # We will store the final eps here:
        self.eps_ = None

        # Standard scaler for data
        self.scaler = StandardScaler()

        # t-SNE will be initialized in fit method since we need to know n_samples
        self.tsne = None
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the DBSCAN algorithm to the data using automatic eps detection via the "knee" method
        and mandatory t-SNE for dimensionality reduction.

        Args:
            data (np.ndarray): An array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels (where -1 indicates noise points).
        """
        n_samples = data.shape[0]
        
        # Set perplexity to min(30, n_samples-1) if not specified
        if self.perplexity is None:
            self.perplexity = min(30, n_samples - 1)
        
        # Initialize t-SNE with appropriate perplexity
        self.tsne = TSNE(
            n_components=self.n_components,
            random_state=self.random_state,
            perplexity=self.perplexity
        )

        # Reduce dimensionality with t-SNE
        X_reduced = self.tsne.fit_transform(data)
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_reduced)
        print("X_scaled", X_scaled.shape)
        # Use NearestNeighbors to find the distances to the n_neighbors
        neighbors = NearestNeighbors(n_neighbors=self.n_neighbors)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, _ = neighbors_fit.kneighbors(X_scaled)

        # Sort distances and pick the distance to the first neighbor (skipping the point itself)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # Determine the knee point for eps
        kneedle = KneeLocator(
            x=range(1, len(distances) + 1),
            y=distances,
            S=self.S,
            curve=self.curve
        )
        eps = kneedle.knee_y
        if eps is None:
            eps = distances[-1]
            logger.warning("Failed to determine eps automatically. Using fallback eps = %f", eps)
        # Create and fit DBSCAN
        self.eps_ = eps  # zapamiętanie wyznaczonego eps
        self.model = DBSCAN(eps=eps, min_samples=self.min_samples)
        labels = self.model.fit_predict(X_scaled)
        self.perplexity = None
        
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print(f"Automatically determined eps: {eps}")
        print(f"DBSCAN: number of clusters = {n_clusters_}, number of noise points = {n_noise_}")
        
        # Plot clusters in 2D using PCA
        try:
            os.makedirs("outputs/clusters", exist_ok=True)
            # Standardize data for PCA
            X_scaled_orig = StandardScaler().fit_transform(data)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled_orig)
            fig, ax = plt.subplots(figsize=(15, 10))
            unique_labels = np.unique(labels)
            # Prepare colormap excluding noise
            if -1 in unique_labels:
                clusters = unique_labels[unique_labels != -1]
            else:
                clusters = unique_labels
            if len(clusters) > 0:
                cmap = plt.cm.get_cmap('viridis', len(clusters))
                for idx, cluster_id in enumerate(clusters):
                    cluster_points = X_pca[labels == cluster_id]
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                               color=cmap(idx), alpha=0.6)
            
            if -1 in unique_labels:
                noise_points = X_pca[labels == -1]
                ax.scatter(noise_points[:, 0], noise_points[:, 1],
                           color='gray', label='Noise', alpha=0.6, marker='x')
            ax.set_title(f'Klasteryzacja DBSCAN (liczba klastrów: {n_clusters_})')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            fig.savefig("outputs/clusters/dbscan_clusters.png", dpi=400)
            plt.close(fig)
        except Exception as e:
            logger.error(f'Failed to plot DBSCAN clusters: {e}')
        
        return labels

    def get_params(self) -> Dict[str, Any]:
        """
        Return the current parameters of the DBSCAN model and the knee method configuration.

        Returns:
            Dict[str, Any]: A dictionary with current parameters.
        """
        return {
            'n_neighbors': self.n_neighbors,
            'min_samples': self.min_samples,
            'n_components': self.n_components,
            'S': self.S,
            'curve': self.curve,
            'perplexity': self.perplexity,
            'eps': self.eps_
        }


# Example of how to implement a new clustering method:
"""
class YourNewClusterer(TimeSeriesClusterer):
    def __init__(self, param1, param2):
        # Initialize your parameters
        self.param1 = param1
        self.param2 = param2
        # Initialize your model
        self.model = YourClusteringAlgorithm(param1=param1, param2=param2)
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        # Implement your clustering logic
        # Remember to handle data preprocessing if needed
        labels = self.model.fit_predict(data)
        return labels
        
    def get_params(self) -> Dict[str, Any]:
        return {
            'param1': self.param1,
            'param2': self.param2
        }
""" 
