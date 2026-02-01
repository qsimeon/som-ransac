"""Utility functions for SOM-based model fitting.

This module provides helper functions for data preprocessing, visualization,
model fitting functions, and evaluation metrics for use with Kohonen SOMs.
"""

import numpy as np
from typing import Tuple, Optional, List, Union


def normalize_data(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """Normalize data for SOM training.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
        normalization_params is a dict containing parameters for denormalization
    """
    if method == 'minmax':
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (data - min_vals) / range_vals
        params = {'method': 'minmax', 'min': min_vals, 'max': max_vals}
        
    elif method == 'zscore':
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        
        # Avoid division by zero
        std_vals[std_vals == 0] = 1.0
        
        normalized = (data - mean_vals) / std_vals
        params = {'method': 'zscore', 'mean': mean_vals, 'std': std_vals}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(data: np.ndarray, params: dict) -> np.ndarray:
    """Denormalize data using stored parameters.
    
    Args:
        data: Normalized data
        params: Dictionary with normalization parameters
        
    Returns:
        Denormalized data
    """
    if params['method'] == 'minmax':
        range_vals = params['max'] - params['min']
        denormalized = data * range_vals + params['min']
        
    elif params['method'] == 'zscore':
        denormalized = data * params['std'] + params['mean']
        
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")
    
    return denormalized


def fit_line_2d(points: np.ndarray) -> dict:
    """Fit a 2D line to points using least squares.
    
    Args:
        points: Array of shape (n_points, 2) with (x, y) coordinates
        
    Returns:
        Dictionary with 'slope', 'intercept', and 'residuals'
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to fit a line")
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Use least squares: y = mx + b
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Calculate residuals
    y_pred = m * x + b
    residuals = np.abs(y - y_pred)
    
    return {
        'slope': m,
        'intercept': b,
        'residuals': residuals,
        'mean_error': np.mean(residuals)
    }


def fit_plane_3d(points: np.ndarray) -> dict:
    """Fit a 3D plane to points using least squares.
    
    Args:
        points: Array of shape (n_points, 3) with (x, y, z) coordinates
        
    Returns:
        Dictionary with plane parameters (a, b, c, d) for ax + by + cz + d = 0
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    
    # Fit plane: z = ax + by + c
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    A = np.column_stack([x, y, np.ones(len(x))])
    coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
    
    a, b, c = coeffs[0], coeffs[1], -1.0
    d = coeffs[2]
    
    # Normalize
    norm = np.sqrt(a**2 + b**2 + c**2)
    a, b, c, d = a/norm, b/norm, c/norm, d/norm
    
    # Calculate residuals
    z_pred = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    residuals = np.abs(z - z_pred)
    
    return {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'residuals': residuals,
        'mean_error': np.mean(residuals)
    }


def fit_circle_2d(points: np.ndarray) -> dict:
    """Fit a 2D circle to points using algebraic fit.
    
    Args:
        points: Array of shape (n_points, 2) with (x, y) coordinates
        
    Returns:
        Dictionary with 'center' (x, y) and 'radius'
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Algebraic circle fit
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    
    center_x = coeffs[0]
    center_y = coeffs[1]
    radius = np.sqrt(coeffs[2] + center_x**2 + center_y**2)
    
    # Calculate residuals
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    residuals = np.abs(distances - radius)
    
    return {
        'center': np.array([center_x, center_y]),
        'radius': radius,
        'residuals': residuals,
        'mean_error': np.mean(residuals)
    }


def generate_synthetic_data(n_inliers: int = 100, n_outliers: int = 20,
                           model_type: str = 'line', noise_std: float = 0.1,
                           random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with inliers and outliers for testing.
    
    Args:
        n_inliers: Number of inlier points
        n_outliers: Number of outlier points
        model_type: Type of model ('line', 'plane', 'circle')
        noise_std: Standard deviation of Gaussian noise for inliers
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (data, ground_truth_labels) where labels are True for inliers
    """
    rng = np.random.RandomState(random_seed)
    
    if model_type == 'line':
        # Generate line: y = 2x + 1
        x = rng.uniform(-5, 5, n_inliers)
        y = 2 * x + 1 + rng.normal(0, noise_std, n_inliers)
        inliers = np.column_stack([x, y])
        
        # Generate outliers
        outliers = rng.uniform(-5, 5, (n_outliers, 2))
        
    elif model_type == 'plane':
        # Generate plane: z = x + 2y + 3
        x = rng.uniform(-5, 5, n_inliers)
        y = rng.uniform(-5, 5, n_inliers)
        z = x + 2*y + 3 + rng.normal(0, noise_std, n_inliers)
        inliers = np.column_stack([x, y, z])
        
        # Generate outliers
        outliers = rng.uniform(-5, 5, (n_outliers, 3))
        
    elif model_type == 'circle':
        # Generate circle: center (0, 0), radius 3
        angles = rng.uniform(0, 2*np.pi, n_inliers)
        radius = 3.0
        x = radius * np.cos(angles) + rng.normal(0, noise_std, n_inliers)
        y = radius * np.sin(angles) + rng.normal(0, noise_std, n_inliers)
        inliers = np.column_stack([x, y])
        
        # Generate outliers
        outliers = rng.uniform(-5, 5, (n_outliers, 2))
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Combine data
    data = np.vstack([inliers, outliers])
    labels = np.concatenate([
        np.ones(n_inliers, dtype=bool),
        np.zeros(n_outliers, dtype=bool)
    ])
    
    # Shuffle
    indices = rng.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    
    return data, labels


def calculate_metrics(predicted_inliers: np.ndarray, 
                     true_inliers: np.ndarray) -> dict:
    """Calculate classification metrics for inlier detection.
    
    Args:
        predicted_inliers: Boolean array of predicted inliers
        true_inliers: Boolean array of ground truth inliers
        
    Returns:
        Dictionary with precision, recall, f1_score, and accuracy
    """
    tp = np.sum(predicted_inliers & true_inliers)
    fp = np.sum(predicted_inliers & ~true_inliers)
    fn = np.sum(~predicted_inliers & true_inliers)
    tn = np.sum(~predicted_inliers & ~true_inliers)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }


def compute_model_error(data: np.ndarray, model_params: dict, 
                       model_type: str) -> np.ndarray:
    """Compute error/distance of points to fitted model.
    
    Args:
        data: Input data points
        model_params: Dictionary with model parameters
        model_type: Type of model ('line', 'plane', 'circle')
        
    Returns:
        Array of errors/distances for each point
    """
    if model_type == 'line':
        x = data[:, 0]
        y = data[:, 1]
        y_pred = model_params['slope'] * x + model_params['intercept']
        errors = np.abs(y - y_pred)
        
    elif model_type == 'plane':
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        # Distance to plane ax + by + cz + d = 0
        a, b, c, d = model_params['a'], model_params['b'], model_params['c'], model_params['d']
        errors = np.abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)
        
    elif model_type == 'circle':
        x = data[:, 0]
        y = data[:, 1]
        center = model_params['center']
        radius = model_params['radius']
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        errors = np.abs(distances - radius)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return errors


def create_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Create pairwise distance matrix for points.
    
    Args:
        points: Array of shape (n_points, n_dims)
        
    Returns:
        Distance matrix of shape (n_points, n_points)
    """
    n = len(points)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(points[i] - points[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def estimate_optimal_map_size(n_samples: int, heuristic: str = 'sqrt') -> Tuple[int, int]:
    """Estimate optimal SOM map size based on data size.
    
    Args:
        n_samples: Number of training samples
        heuristic: Heuristic to use ('sqrt', 'log', 'linear')
        
    Returns:
        Tuple of (rows, cols) for map size
    """
    if heuristic == 'sqrt':
        side = int(np.sqrt(5 * np.sqrt(n_samples)))
    elif heuristic == 'log':
        side = int(np.log(n_samples) * 2)
    elif heuristic == 'linear':
        side = int(n_samples ** 0.4)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
    
    side = max(5, min(side, 50))  # Clamp between 5 and 50
    
    return (side, side)
