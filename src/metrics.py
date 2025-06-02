"""
Custom metrics for model evaluation.
"""

import tensorflow as tf
import numpy as np


def root_mean_squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def mean_absolute_percentage_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value
    """
    epsilon = tf.keras.backend.epsilon()  # Small constant to avoid division by zero
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon))
    return 100.0 * tf.reduce_mean(diff)


def symmetric_mean_absolute_percentage_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        sMAPE value
    """
    epsilon = tf.keras.backend.epsilon()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0 + epsilon
    diff = tf.abs(y_true - y_pred) / denominator
    return 100.0 * tf.reduce_mean(diff)


def mean_directional_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate Mean Directional Accuracy (MDA).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MDA value
    """
    # Calculate directions (1 for increase, 0 for decrease)
    y_true_direction = tf.cast(y_true[:, 1:] > y_true[:, :-1], tf.float32)
    y_pred_direction = tf.cast(y_pred[:, 1:] > y_pred[:, :-1], tf.float32)
    
    # Calculate accuracy of direction prediction
    correct_direction = tf.cast(y_true_direction == y_pred_direction, tf.float32)
    return 100.0 * tf.reduce_mean(correct_direction)


def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R-squared value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate total sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    
    # Calculate residual sum of squares
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    
    # Calculate R-squared
    r2 = 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())
    
    return r2


def weighted_mean_absolute_error(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor = None) -> tf.Tensor:
    """
    Calculate Weighted Mean Absolute Error (WMAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        weights: Optional weights for each sample
        
    Returns:
        WMAE value
    """
    if weights is None:
        # If no weights provided, use uniform weights
        weights = tf.ones_like(y_true)
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    weights = tf.cast(weights, tf.float32)
    
    # Normalize weights
    weights = weights / tf.reduce_sum(weights)
    
    # Calculate weighted MAE
    return tf.reduce_sum(weights * tf.abs(y_true - y_pred)) 