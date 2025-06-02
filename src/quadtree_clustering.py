import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, List, Optional
from scipy.spatial import cKDTree
from .clustering import TimeSeriesClusterer


class QuadTreeClusterer(TimeSeriesClusterer):
    """
    QuadTree-based spatial clustering implementation.
    
    This clusterer uses recursive spatial subdivision to partition spatial data into clusters
    based on geographical proximity. It's suitable for large datasets where spatial
    distribution is an important factor.
    
    Example usage:
    ```python
    # Initialize clusterer
    quad_clusterer = QuadTreeClusterer(max_points_per_leaf=50, min_points_per_leaf=10, max_depth=10)
    
    # Prepare your data with latitude and longitude columns
    geo_data = np.array([...])  # Your geo-coordinates data (longitude, latitude)
    
    # Fit and get cluster labels
    labels = quad_clusterer.fit(geo_data)
    ```
    """
    
    def __init__(
        self,
        max_points_per_leaf: int = 30,
        min_points_per_leaf: int = 10,
        max_depth: int = 10,
        source_crs: str = "EPSG:4326",
        target_crs: str = "EPSG:2177"
    ):
        """
        Initialize QuadTree clusterer.
        
        Args:
            max_points_per_leaf: Maximum number of points in a leaf node
            min_points_per_leaf: Minimum number of points in a leaf node
            max_depth: Maximum depth of the QuadTree
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system for spatial operations
        """
        self.max_points_per_leaf = max_points_per_leaf
        self.min_points_per_leaf = min_points_per_leaf
        self.max_depth = max_depth
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.leaf_id_map = {}
        self.leaf_stats = []
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit QuadTree clustering to the data.
        
        Args:
            data: Array of shape (n_samples, n_features) where the first two columns
                 are assumed to be longitude and latitude, respectively.
            
        Returns:
            np.ndarray: Cluster labels (quad_id for each point)
        """
        if data.shape[1] < 2:
            raise ValueError("Data must have at least 2 columns (longitude, latitude)")
        
        print(f"Total number of points: {len(data)}")
        
        # Extract longitude and latitude columns
        lon_lat_data = data[:, :2]
        
        # Create GeoDataFrame with Point geometry
        geometry = [Point(lon, lat) for lon, lat in lon_lat_data]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs=self.source_crs)
        
        # Transform to target CRS
        gdf_transformed = gdf.to_crs(self.target_crs)
        
        # Extract coordinates in target CRS
        x_coords = np.array([geom.x for geom in gdf_transformed.geometry])
        y_coords = np.array([geom.y for geom in gdf_transformed.geometry])
        
        # Build point array
        points = np.column_stack((x_coords, y_coords))
        
        # Reset stats and ID map
        self.leaf_id_map = {}
        self.leaf_stats = []
        
        # Start recursive quad division
        cluster_id = 0
        labels = np.zeros(len(data), dtype=int) - 1  # Initialize with -1 (unclustered)
        
        # Find the bounding box for all points
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Create indices array for all points
        all_indices = np.arange(len(points))
        
        # Create and build the quad tree recursively
        cluster_id = self._build_quad_tree(
            points, all_indices, labels, cluster_id, 
            x_min, y_min, x_max, y_max, 0  # Start at depth 0
        )
        
        print(f"Number of clusters: {cluster_id}")
        
        # Save quad_stats as DataFrame
        self.quad_stats_df = pd.DataFrame(self.leaf_stats)
        print("\nCluster statistics:")
        print(self.quad_stats_df[['cluster_id', 'point_count', 'depth']])
        
        return labels
    
    def _build_quad_tree(self, points, indices, labels, cluster_id, 
                        x_min, y_min, x_max, y_max, depth):
        """
        Recursively build a quad tree and assign cluster labels.
        
        Args:
            points: Array of point coordinates
            indices: Array of point indices within the current quad
            labels: Array of cluster labels (modified in-place)
            cluster_id: Current cluster ID
            x_min, y_min, x_max, y_max: Bounding box of current quad
            depth: Current depth in the quad tree
            
        Returns:
            Next available cluster ID
        """
        # Count points in this quad
        n_points = len(indices)
        
        # If we've reached max depth or have few enough points, create a leaf node
        if depth >= self.max_depth or n_points <= self.max_points_per_leaf:
            # Only create a cluster if we have at least min_points_per_leaf
            if n_points >= self.min_points_per_leaf:
                # Assign cluster ID to all points in this quad
                labels[indices] = cluster_id
                
                # Create polygon for visualization
                polygon = Polygon([
                    (x_min, y_min), (x_max, y_min),
                    (x_max, y_max), (x_min, y_max)
                ])
                
                # Store leaf stats
                self.leaf_stats.append({
                    'cluster_id': cluster_id,
                    'point_count': n_points,
                    'depth': depth,
                    'geometry': polygon,
                    'area': polygon.area
                })
                
                # Increment cluster ID for next cluster
                cluster_id += 1
            return cluster_id
        
        # If we have more points than max_points_per_leaf, divide the quad
        # Calculate midpoints
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        
        # Create four quads
        quads = [
            # bottom-left
            (x_min, y_min, x_mid, y_mid),
            # bottom-right
            (x_mid, y_min, x_max, y_mid),
            # top-left
            (x_min, y_mid, x_mid, y_max),
            # top-right
            (x_mid, y_mid, x_max, y_max)
        ]
        
        # Assign points to quads
        for quad in quads:
            qx_min, qy_min, qx_max, qy_max = quad
            
            # Find points in this quad
            mask = (
                (points[indices, 0] >= qx_min) & 
                (points[indices, 0] < qx_max) & 
                (points[indices, 1] >= qy_min) & 
                (points[indices, 1] < qy_max)
            )
            
            # Get indices of points in this quad
            quad_indices = indices[mask]
            
            # If we have points in this quad, recursively process it
            if len(quad_indices) > 0:
                cluster_id = self._build_quad_tree(
                    points, quad_indices, labels, cluster_id,
                    qx_min, qy_min, qx_max, qy_max, depth + 1
                )
        
        return cluster_id
    
    def get_params(self) -> Dict:
        """
        Get the parameters of the clustering algorithm.
        
        Returns:
            Dict containing the algorithm parameters
        """
        return {
            'max_points_per_leaf': self.max_points_per_leaf,
            'min_points_per_leaf': self.min_points_per_leaf,
            'max_depth': self.max_depth,
            'source_crs': self.source_crs,
            'target_crs': self.target_crs
        }
        
    def plot_clusters(self, output_dir: str = "./outputs", filename: str = "quadtree_clusters.png"):
        """
        Plot QuadTree clusters.
        
        Args:
            output_dir: Directory to save the plot
            filename: Filename for the plot
        """
        if not self.leaf_stats:
            raise ValueError("No clustering has been performed yet. Call fit() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create GeoDataFrame from leaf stats
        quad_gdf = gpd.GeoDataFrame(self.quad_stats_df, geometry='geometry', crs=self.target_crs)
        
        # Plot
        fig, ax = plt.subplots(figsize=(15, 12))
        quad_gdf.plot(
            ax=ax,
            column='point_count',
            cmap='viridis',
            legend=True,
            alpha=0.6,
            edgecolor='black'
        )
        
        plt.title(f'QuadTree Clustering (min: {self.min_points_per_leaf}, max: {self.max_points_per_leaf}, depth: {self.max_depth})')
        plt.xlabel(f'X ({self.target_crs})')
        plt.ylabel(f'Y ({self.target_crs})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        
        return fig, ax
    

    
def quadtree_clustering(
    geo_df: pd.DataFrame,
    max_points_per_leaf: int = 30,
    min_points_per_leaf: int = 10,
    max_depth: int = 20,
    plot: bool = True,
    output_dir: str = "./outputs"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform QuadTree-based spatial clustering on geographical data.
    
    Parameters:
    -----------
    geo_df : DataFrame
        DataFrame with geographical data (must contain 'latitude' and 'longitude' columns)
    max_points_per_leaf : int
        Maximum number of points per leaf in the QuadTree
    min_points_per_leaf : int
        Minimum number of points per leaf in the QuadTree
    max_depth : int
        Maximum depth of the QuadTree
    plot : bool
        Whether to generate a plot of the clustering
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    geo_df : DataFrame
        DataFrame with added 'quad_id' column
    quad_stats : DataFrame
        DataFrame with statistics for each QuadTree leaf
    """
    # Initialize the clusterer
    clusterer = QuadTreeClusterer(
        max_points_per_leaf=max_points_per_leaf,
        min_points_per_leaf=min_points_per_leaf,
        max_depth=max_depth
    )
    
    # Prepare the data for clustering
    cluster_input = np.column_stack((
        geo_df['longitude'].values,
        geo_df['latitude'].values
    ))
    
    # Perform clustering
    labels = clusterer.fit(cluster_input)
    
    # Add cluster IDs to the dataframe
    geo_df = geo_df.copy()
    geo_df['quad_id'] = labels
    
    # Generate plot if requested
    if plot:
        clusterer.plot_clusters(output_dir=output_dir)
    
    return geo_df, clusterer.quad_stats_df 