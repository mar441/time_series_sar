"""
Visualization module for time series forecasting results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime
from pathlib import Path


def set_plotting_style():
    """Set consistent style for all matplotlib plots."""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2


def plot_loss_interactive(
    history_dict: Dict[str, Dict[str, List[float]]],
    output_dir: str,
    model_type: str
):
    """
    Create interactive loss plot using plotly.
    
    Args:
        history_dict: Dictionary of training histories
        output_dir: Directory to save plot
        model_type: Type of model used
    """
    fig = go.Figure()
    
    for param_set, history in history_dict.items():
        # Training loss
        fig.add_trace(go.Scatter(
            y=history['loss'],
            name=f'Training Loss - {param_set}',
            mode='lines',
            line=dict(width=2, dash='solid')
        ))
        
        # Validation loss
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                name=f'Validation Loss - {param_set}',
                mode='lines',
                line=dict(width=2, dash='dash')
            ))
    
    fig.update_layout(
        title=f'{model_type} Model Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save interactive plot
    output_path = os.path.join(output_dir, f'{model_type}_loss_interactive.html')
    fig.write_html(output_path)


def plot_predictions_interactive(
    actual_data: pd.DataFrame,
    predictions: pd.DataFrame,
    point_name: str,
    output_dir: str
):
    """
    Create interactive prediction plot using plotly.
    
    Args:
        actual_data: DataFrame with actual values
        predictions: DataFrame with predicted values
        point_name: Name of the point being plotted
        output_dir: Directory to save plot
    """
    fig = go.Figure()
    
    # Plot actual values
    fig.add_trace(go.Scatter(
        x=actual_data.index,
        y=actual_data[point_name],
        name='Actual',
        mode='lines',
        line=dict(color='blue', width=2)
    ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions[point_name],
        name='Predicted',
        mode='lines',
        line=dict(color='red', width=2)
    ))
    
    # Add confidence intervals if available
    if f'{point_name}_lower' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions[f'{point_name}_lower'],
            name='Lower Bound',
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions[f'{point_name}_upper'],
            name='Upper Bound',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'Time Series Predictions for {point_name}',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save interactive plot
    output_path = os.path.join(output_dir, f'{point_name}_predictions_interactive.html')
    fig.write_html(output_path)


def plot_error_distribution(
    actual: np.ndarray,
    predicted: np.ndarray,
    point_name: str,
    output_dir: str
):
    """
    Plot error distribution using both histogram and KDE.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        point_name: Name of the point being analyzed
        output_dir: Directory to save plot
    """
    errors = predicted - actual
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=errors,
            name="Error Distribution",
            nbinsx=50,
            opacity=0.7
        ),
        secondary_y=False,
    )
    
    # Add KDE
    kde_x = np.linspace(min(errors), max(errors), 100)
    kde = sns.kde(errors)
    kde_y = kde.get_density(kde_x)
    
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y,
            name="KDE",
            line=dict(color='red', width=2)
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=f'Error Distribution for {point_name}',
        xaxis_title='Error',
        yaxis_title='Count',
        yaxis2_title='Density',
        template='plotly_white',
        showlegend=True
    )
    
    # Save interactive plot
    output_path = os.path.join(output_dir, f'{point_name}_error_distribution.html')
    fig.write_html(output_path)


def plot_feature_importance(
    importance_scores: Dict[str, float],
    point_name: str,
    output_dir: str
):
    """
    Create interactive feature importance plot.
    
    Args:
        importance_scores: Dictionary of feature importance scores
        point_name: Name of the point being analyzed
        output_dir: Directory to save plot
    """
    # Sort features by importance
    sorted_features = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    features, scores = zip(*sorted_features)
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation='h'
    ))
    
    fig.update_layout(
        title=f'Feature Importance for {point_name}',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_white',
        height=max(400, len(features) * 20)  # Adjust height based on number of features
    )
    
    # Save interactive plot
    output_path = os.path.join(output_dir, f'{point_name}_feature_importance.html')
    fig.write_html(output_path)


def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    output_dir: str,
    min_correlation: float = 0.5
):
    """
    Create interactive correlation matrix plot.
    
    Args:
        correlation_matrix: DataFrame with correlation values
        output_dir: Directory to save plot
        min_correlation: Minimum correlation to display
    """
    # Filter correlations
    mask = np.abs(correlation_matrix) >= min_correlation
    correlation_matrix = correlation_matrix.where(mask)
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Point Correlation Matrix',
        template='plotly_white',
        height=800,
        width=800
    )
    
    # Save interactive plot
    output_path = os.path.join(output_dir, 'correlation_matrix.html')
    fig.write_html(output_path)


def plot_cluster_analysis(
    data: pd.DataFrame,
    clusters: np.ndarray,
    representative_points: List[str],
    output_dir: str
):
    """
    Create interactive cluster analysis visualization.
    
    Args:
        data: DataFrame with time series data
        clusters: Array of cluster assignments
        representative_points: List of representative point names
        output_dir: Directory to save plot
    """
    # Perform PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data.T)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': data_2d[:, 0],
        'PC2': data_2d[:, 1],
        'Cluster': clusters,
        'Point': data.columns,
        'Is_Representative': data.columns.isin(representative_points)
    })
    
    fig = go.Figure()
    
    # Plot points for each cluster
    for cluster_id in np.unique(clusters):
        cluster_data = plot_df[plot_df['Cluster'] == cluster_id]
        
        # Regular points
        regular_points = cluster_data[~cluster_data['Is_Representative']]
        fig.add_trace(go.Scatter(
            x=regular_points['PC1'],
            y=regular_points['PC2'],
            mode='markers',
            name=f'Cluster {cluster_id}',
            text=regular_points['Point'],
            marker=dict(size=8)
        ))
        
        # Representative points
        rep_points = cluster_data[cluster_data['Is_Representative']]
        if not rep_points.empty:
            fig.add_trace(go.Scatter(
                x=rep_points['PC1'],
                y=rep_points['PC2'],
                mode='markers',
                name=f'Representative (Cluster {cluster_id})',
                text=rep_points['Point'],
                marker=dict(
                    size=15,
                    symbol='star',
                    line=dict(width=2)
                )
            ))
    
    fig.update_layout(
        title='Cluster Analysis of Points',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        template='plotly_white',
        showlegend=True,
        height=800,
        width=1000
    )
    
    # Save interactive plot
    output_path = os.path.join(output_dir, 'cluster_analysis.html')
    fig.write_html(output_path)


def create_dashboard(
    actual_data: pd.DataFrame,
    predictions: pd.DataFrame,
    history_dict: Dict[str, Dict[str, List[float]]],
    model_type: str,
    output_dir: str
):
    """
    Create comprehensive interactive dashboard.
    
    Args:
        actual_data: DataFrame with actual values
        predictions: DataFrame with predicted values
        history_dict: Dictionary of training histories
        model_type: Type of model used
        output_dir: Directory to save dashboard
    """
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training History',
            'Predictions Overview',
            'Error Distribution',
            'Performance Metrics'
        )
    )
    
    # Add training history
    for param_set, history in history_dict.items():
        fig.add_trace(
            go.Scatter(
                y=history['loss'],
                name=f'Training Loss - {param_set}',
                mode='lines'
            ),
            row=1, col=1
        )
    
    # Add predictions overview
    for column in predictions.columns:
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions[column],
                name=f'Predictions - {column}',
                mode='lines'
            ),
            row=1, col=2
        )
    
    # Add error distribution
    errors = []
    for column in predictions.columns:
        if column in actual_data.columns:
            error = predictions[column] - actual_data[column]
            errors.extend(error.dropna())
    
    fig.add_trace(
        go.Histogram(
            x=errors,
            name='Error Distribution',
            nbinsx=50
        ),
        row=2, col=1
    )
    
    # Add performance metrics
    metrics = calculate_performance_metrics(actual_data, predictions)
    fig.add_trace(
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            name='Performance Metrics'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=1000,
        width=1500,
        title_text=f"{model_type} Model Dashboard",
        showlegend=True,
        template='plotly_white'
    )
    
    # Save dashboard
    output_path = os.path.join(output_dir, f'{model_type}_dashboard.html')
    fig.write_html(output_path)


def calculate_performance_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate various performance metrics.
    
    Args:
        actual: DataFrame with actual values
        predicted: DataFrame with predicted values
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    for column in predicted.columns:
        if column in actual.columns:
            y_true = actual[column].dropna()
            y_pred = predicted[column].dropna()
        
        # Calculate metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            
            # Calculate MAPE avoiding division by zero
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
            metrics[f'{column}_MSE'] = mse
            metrics[f'{column}_RMSE'] = rmse
            metrics[f'{column}_MAE'] = mae
            metrics[f'{column}_MAPE'] = mape
    
    return metrics 


def plot_sequence_prediction_comparison(
    actual_sequences: np.ndarray,
    predicted_sequences: np.ndarray,
    point_indices: np.ndarray,
    point_names: List[str],
    output_dir: str,
    max_sequences: int = 5,
    show_confidence: bool = True,
    confidence_interval: float = 0.95
):
    """
    Create interactive plots comparing actual and predicted sequences.
    
    Args:
        actual_sequences: Array of actual sequences
        predicted_sequences: Array of predicted sequences
        point_indices: Array of point indices
        point_names: List of point names
        output_dir: Directory to save plots
        max_sequences: Maximum number of sequences to plot per point
        show_confidence: Whether to show confidence intervals
        confidence_interval: Confidence interval level (0 to 1)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group sequences by point
    unique_points = np.unique(point_indices)
    
    for point_idx in unique_points:
        point_mask = point_indices == point_idx
        point_actuals = actual_sequences[point_mask]
        point_preds = predicted_sequences[point_mask]
        
        # Select random sequences if there are too many
        if len(point_actuals) > max_sequences:
            indices = np.random.choice(
                len(point_actuals),
                size=max_sequences,
                replace=False
            )
            point_actuals = point_actuals[indices]
            point_preds = point_preds[indices]
        
        fig = go.Figure()
        
        # Plot each sequence
        for i, (actual, pred) in enumerate(zip(point_actuals, point_preds)):
            # Actual sequence
            fig.add_trace(go.Scatter(
                y=actual,
                mode='lines',
                name=f'Actual {i+1}',
                line=dict(color='blue', width=2, dash='solid')
            ))
            
            # Predicted sequence
            fig.add_trace(go.Scatter(
                y=pred,
                mode='lines',
                name=f'Predicted {i+1}',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add confidence intervals if requested
            if show_confidence:
                std_dev = np.std(actual - pred)
                z_score = 1.96  # for 95% confidence interval
                margin = z_score * std_dev
                
                fig.add_trace(go.Scatter(
                    y=pred + margin,
                    mode='lines',
                    name=f'Upper CI {i+1}',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    y=pred - margin,
                    mode='lines',
                    name=f'Lower CI {i+1}',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(width=0),
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=f'Sequence Predictions for {point_names[point_idx]}',
            xaxis_title='Time Step',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save plot
        output_path = os.path.join(
            output_dir,
            f'sequence_comparison_{point_names[point_idx]}.html'
        )
        fig.write_html(output_path)


def plot_predictions(
    actual_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    point_names: List[str],
    output_dir: str,
    plot_type: str = 'interactive',
    show_confidence: bool = True,
    confidence_interval: float = 0.95
):
    """
    Plot predictions against actual values.
    
    Args:
        actual_data: DataFrame with actual values
        predicted_data: DataFrame with predicted values
        point_names: List of point names to plot
        output_dir: Directory to save plots
        plot_type: Type of plot ('interactive' or 'static')
        show_confidence: Whether to show confidence intervals
        confidence_interval: Confidence interval level (0 to 1)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for point in point_names:
        if plot_type == 'interactive':
            fig = go.Figure()
            
            # Plot actual values
            fig.add_trace(go.Scatter(
                x=actual_data.index,
                y=actual_data[point],
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            
            # Plot predictions
            fig.add_trace(go.Scatter(
                x=predicted_data.index,
                y=predicted_data[point],
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2)
            ))
            
            # Add confidence intervals if requested
            if show_confidence:
                errors = actual_data[point] - predicted_data[point]
                std_dev = np.std(errors.dropna())
                z_score = 1.96  # for 95% confidence interval
                margin = z_score * std_dev
                
                fig.add_trace(go.Scatter(
                    x=predicted_data.index,
                    y=predicted_data[point] + margin,
                    mode='lines',
                    name='Upper CI',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=predicted_data.index,
                    y=predicted_data[point] - margin,
                    mode='lines',
                    name='Lower CI',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(width=0),
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Predictions vs Actual for {point}',
                xaxis_title='Time',
                yaxis_title='Value',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Save plot
            output_path = os.path.join(output_dir, f'predictions_{point}.html')
            fig.write_html(output_path)
            
        else:  # static plot
            plt.figure(figsize=(12, 6))
            plt.plot(actual_data.index, actual_data[point], 'b-', label='Actual')
            plt.plot(predicted_data.index, predicted_data[point], 'r--', label='Predicted')
            
            if show_confidence:
                errors = actual_data[point] - predicted_data[point]
                std_dev = np.std(errors.dropna())
                z_score = 1.96
                margin = z_score * std_dev
                
                plt.fill_between(
                    predicted_data.index,
                    predicted_data[point] - margin,
                    predicted_data[point] + margin,
                    color='r',
                    alpha=0.1,
                    label=f'{int(confidence_interval*100)}% CI'
                )
            
            plt.title(f'Predictions vs Actual for {point}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            output_path = os.path.join(output_dir, f'predictions_{point}.png')
            plt.savefig(output_path)
            plt.close() 


def plot_loss(
    history: Dict[str, List[float]],
    output_dir: str,
    model_name: str = 'model',
    plot_type: str = 'interactive'
):
    """
    Plot training and validation loss.
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plot
        model_name: Name of the model
        plot_type: Type of plot ('interactive' or 'static')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle empty or invalid history
    if not history or not isinstance(history, dict):
        return
        
    # Get available metrics
    metrics = []
    if 'loss' in history:
        metrics.append(('loss', 'Training Loss'))
    if 'val_loss' in history:
        metrics.append(('val_loss', 'Validation Loss'))
    
    # For ML models that might have different metric names
    if not metrics and len(history) > 0:
        # Use the first available metric
        metric_name = list(history.keys())[0]
        metrics.append((metric_name, 'Model Loss'))
    
    if plot_type == 'interactive':
        fig = go.Figure()
        
        # Plot each metric
        for metric_key, metric_label in metrics:
            fig.add_trace(go.Scatter(
                y=history[metric_key],
                name=metric_label,
                mode='lines',
                line=dict(width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{model_name} Training History',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save plot
        output_path = os.path.join(output_dir, f'{model_name}_loss.html')
        fig.write_html(output_path)
        
    else:  # static plot
        plt.figure(figsize=(10, 6))
        
        # Plot each metric
        for metric_key, metric_label in metrics:
            plt.plot(history[metric_key], label=metric_label)
        
        plt.title(f'{model_name} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_path = os.path.join(output_dir, f'{model_name}_loss.png')
        plt.savefig(output_path)
        plt.close() 


def plot_input_sequences(
    sequences: np.ndarray,
    point_indices: np.ndarray,
    point_names: List[str],
    output_dir: str,
    max_sequences: int = 5,
    plot_type: str = 'interactive'
):
    """
    Plot input sequences for each point.
    
    Args:
        sequences: Array of input sequences
        point_indices: Array of point indices
        point_names: List of point names
        output_dir: Directory to save plots
        max_sequences: Maximum number of sequences to plot per point
        plot_type: Type of plot ('interactive' or 'static')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group sequences by point
    unique_points = np.unique(point_indices)
    
    for point_idx in unique_points:
        point_mask = point_indices == point_idx
        point_sequences = sequences[point_mask]
        
        # Select random sequences if there are too many
        if len(point_sequences) > max_sequences:
            indices = np.random.choice(
                len(point_sequences),
                size=max_sequences,
                replace=False
            )
            point_sequences = point_sequences[indices]
        
        if plot_type == 'interactive':
            fig = go.Figure()
            
            # Plot each sequence
            for i, sequence in enumerate(point_sequences):
                fig.add_trace(go.Scatter(
                    y=sequence.flatten(),
                    mode='lines',
                    name=f'Sequence {i+1}',
                    line=dict(width=2)
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Input Sequences for {point_names[point_idx]}',
                xaxis_title='Time Step',
                yaxis_title='Value',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
        
        # Save plot
            output_path = os.path.join(
                output_dir,
                f'input_sequences_{point_names[point_idx]}.html'
            )
            fig.write_html(output_path)
            
        else:  # static plot
            plt.figure(figsize=(12, 6))
            
            for sequence in point_sequences:
                plt.plot(sequence.flatten(), alpha=0.7)
            
            plt.title(f'Input Sequences for {point_names[point_idx]}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.grid(True)
    
    # Save plot
            output_path = os.path.join(
                output_dir,
                f'input_sequences_{point_names[point_idx]}.png'
            )
            plt.savefig(output_path)
    plt.close() 


def plot_model_comparison(
    actual_data: pd.DataFrame,
    predictions_dict: Dict[str, pd.DataFrame],
    point_names: List[str],
    output_dir: str,
    show_confidence: bool = True,
    confidence_interval: float = 0.95
):
    """
    Create comparative plots showing predictions from all models against actual data.
    
    Args:
        actual_data: DataFrame with actual values
        predictions_dict: Dictionary mapping model names to their prediction DataFrames
        point_names: List of point names to plot
        output_dir: Directory to save plots
        show_confidence: Whether to show confidence intervals
        confidence_interval: Confidence interval level (default: 0.95)
    """
    # Create comparison plots directory
    comparison_dir = os.path.join(output_dir, "model_comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Color map for different models with their fill colors
    model_colors = {
        'lstm': {'line': 'red', 'fill': 'rgba(255, 0, 0, 0.2)'},
        'conv_autoencoder': {'line': 'green', 'fill': 'rgba(0, 255, 0, 0.2)'},
        'dense_autoencoder': {'line': 'blue', 'fill': 'rgba(0, 0, 255, 0.2)'},
        'ml': {'line': 'purple', 'fill': 'rgba(128, 0, 128, 0.2)'}
    }
    
    # Create plot for each point
    for point in point_names:
        fig = go.Figure()
        
        # Plot actual data
        fig.add_trace(go.Scatter(
            x=actual_data.index,
            y=actual_data[point],
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Plot predictions from each model
        for model_name, predictions_df in predictions_dict.items():
            if point in predictions_df.columns:
                model_color = model_colors.get(model_name.lower(), {'line': 'gray', 'fill': 'rgba(128, 128, 128, 0.2)'})
                
                # Add model predictions
                fig.add_trace(go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df[point],
                    mode='lines',
                    name=f'{model_name.upper()}',
                    line=dict(
                        color=model_color['line'],
                        width=2,
                        dash='dash'
                    )
                ))
                
                # Add confidence intervals if requested
                if show_confidence:
                    errors = actual_data[point] - predictions_df[point]
                    std_dev = np.std(errors.dropna())
                    z_score = 1.96  # for 95% confidence interval
                    margin = z_score * std_dev
                    
                    fig.add_trace(go.Scatter(
                        x=predictions_df.index,
                        y=predictions_df[point] + margin,
                        mode='lines',
                        name=f'{model_name} Upper CI',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=predictions_df.index,
                        y=predictions_df[point] - margin,
                        mode='lines',
                        name=f'{model_name} Lower CI',
                        fill='tonexty',
                        fillcolor=model_color['fill'],
                        line=dict(width=0),
                        showlegend=False
                    ))
        
        # Update layout
        fig.update_layout(
            title=f'Model Comparison for {point}',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save plot
        output_path = os.path.join(comparison_dir, f'model_comparison_{point}.html')
        fig.write_html(output_path)
        
        # # Also save as static image for quick viewing
        # output_path_png = os.path.join(comparison_dir, f'model_comparison_{point}.png')
        # fig.write_image(output_path_png)