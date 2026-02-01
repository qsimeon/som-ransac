"""Core module for Kohonen Self-Organizing Maps (SOM) as an alternative to RANSAC.

This module provides the main SOM implementation for robust model fitting,
offering an alternative approach to RANSAC for outlier rejection and
parameter estimation in computer vision and data fitting tasks.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class SOMConfig:
    """Configuration for Self-Organizing Map.
    
    Attributes:
        map_size: Tuple of (rows, cols) for the SOM grid
        input_dim: Dimensionality of input vectors
        learning_rate_init: Initial learning rate
        learning_rate_final: Final learning rate
        sigma_init: Initial neighborhood radius
        sigma_final: Final neighborhood radius
        max_iterations: Maximum number of training iterations
        random_seed: Random seed for reproducibility
    """
    map_size: Tuple[int, int] = (10, 10)
    input_dim: int = 2
    learning_rate_init: float = 0.5
    learning_rate_final: float = 0.01
    sigma_init: float = None  # Will default to max(map_size) / 2
    sigma_final: float = 0.5
    max_iterations: int = 1000
    random_seed: Optional[int] = None


class KohonenSOM:
    """Kohonen Self-Organizing Map for robust model fitting.
    
    This class implements a SOM that can be used as an alternative to RANSAC
    for tasks like line fitting, plane fitting, and other geometric model
    estimation problems with outliers.
    """
    
    def __init__(self, config: SOMConfig):
        """Initialize the Kohonen SOM.
        
        Args:
            config: SOMConfig object with network parameters
        """
        self.config = config
        self.map_size = config.map_size
        self.input_dim = config.input_dim
        
        # Set default sigma_init if not provided
        if config.sigma_init is None:
            self.config.sigma_init = max(self.map_size) / 2.0
        
        # Initialize random state
        self.rng = np.random.RandomState(config.random_seed)
        
        # Initialize weight matrix: (rows, cols, input_dim)
        self.weights = self.rng.randn(
            self.map_size[0], 
            self.map_size[1], 
            self.input_dim
        )
        
        # Create coordinate grid for neurons
        self.neuron_coords = self._create_neuron_grid()
        
        # Training state
        self.iteration = 0
        self.is_trained = False
    
    def _create_neuron_grid(self) -> np.ndarray:
        """Create coordinate grid for SOM neurons.
        
        Returns:
            Array of shape (rows, cols, 2) with neuron coordinates
        """
        rows, cols = self.map_size
        coords = np.zeros((rows, cols, 2))
        for i in range(rows):
            for j in range(cols):
                coords[i, j] = [i, j]
        return coords
    
    def _get_learning_rate(self, iteration: int) -> float:
        """Calculate learning rate for current iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Learning rate value
        """
        progress = iteration / self.config.max_iterations
        lr = self.config.learning_rate_init * (
            (self.config.learning_rate_final / self.config.learning_rate_init) ** progress
        )
        return lr
    
    def _get_sigma(self, iteration: int) -> float:
        """Calculate neighborhood radius for current iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Sigma (neighborhood radius) value
        """
        progress = iteration / self.config.max_iterations
        sigma = self.config.sigma_init * (
            (self.config.sigma_final / self.config.sigma_init) ** progress
        )
        return sigma
    
    def _find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit (BMU) for input vector.
        
        Args:
            input_vector: Input data point of shape (input_dim,)
            
        Returns:
            Tuple of (row, col) indices of BMU
        """
        # Calculate Euclidean distance to all neurons
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def _update_weights(self, input_vector: np.ndarray, bmu_idx: Tuple[int, int], 
                       learning_rate: float, sigma: float) -> None:
        """Update weights using SOM learning rule.
        
        Args:
            input_vector: Input data point
            bmu_idx: Best Matching Unit indices
            learning_rate: Current learning rate
            sigma: Current neighborhood radius
        """
        bmu_coord = self.neuron_coords[bmu_idx[0], bmu_idx[1]]
        
        # Calculate distances from BMU to all neurons
        distances_sq = np.sum(
            (self.neuron_coords - bmu_coord) ** 2, axis=2
        )
        
        # Calculate neighborhood function (Gaussian)
        neighborhood = np.exp(-distances_sq / (2 * sigma ** 2))
        
        # Expand dimensions for broadcasting
        neighborhood = neighborhood[:, :, np.newaxis]
        
        # Update weights
        self.weights += learning_rate * neighborhood * (
            input_vector - self.weights
        )
    
    def fit(self, data: np.ndarray, verbose: bool = False) -> 'KohonenSOM':
        """Train the SOM on input data.
        
        Args:
            data: Training data of shape (n_samples, input_dim)
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        n_samples = data.shape[0]
        
        for iteration in range(self.config.max_iterations):
            # Get current hyperparameters
            lr = self._get_learning_rate(iteration)
            sigma = self._get_sigma(iteration)
            
            # Select random sample
            idx = self.rng.randint(0, n_samples)
            input_vector = data[idx]
            
            # Find BMU and update weights
            bmu_idx = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_idx, lr, sigma)
            
            self.iteration = iteration
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.config.max_iterations}, "
                      f"LR: {lr:.4f}, Sigma: {sigma:.4f}")
        
        self.is_trained = True
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Find BMU indices for input data.
        
        Args:
            data: Input data of shape (n_samples, input_dim)
            
        Returns:
            Array of shape (n_samples, 2) with BMU coordinates
        """
        bmu_indices = np.zeros((data.shape[0], 2), dtype=int)
        for i, sample in enumerate(data):
            bmu_idx = self._find_bmu(sample)
            bmu_indices[i] = bmu_idx
        return bmu_indices
    
    def get_quantization_error(self, data: np.ndarray) -> float:
        """Calculate average quantization error.
        
        Args:
            data: Input data of shape (n_samples, input_dim)
            
        Returns:
            Average quantization error
        """
        errors = []
        for sample in data:
            bmu_idx = self._find_bmu(sample)
            bmu_weight = self.weights[bmu_idx[0], bmu_idx[1]]
            error = np.linalg.norm(sample - bmu_weight)
            errors.append(error)
        return np.mean(errors)
    
    def detect_outliers(self, data: np.ndarray, threshold: Optional[float] = None,
                       percentile: float = 90) -> np.ndarray:
        """Detect outliers based on quantization error.
        
        Args:
            data: Input data of shape (n_samples, input_dim)
            threshold: Manual threshold for outlier detection (optional)
            percentile: Percentile for automatic threshold if threshold is None
            
        Returns:
            Boolean array where True indicates inlier, False indicates outlier
        """
        errors = []
        for sample in data:
            bmu_idx = self._find_bmu(sample)
            bmu_weight = self.weights[bmu_idx[0], bmu_idx[1]]
            error = np.linalg.norm(sample - bmu_weight)
            errors.append(error)
        
        errors = np.array(errors)
        
        if threshold is None:
            threshold = np.percentile(errors, percentile)
        
        inliers = errors <= threshold
        return inliers


class SOMModelFitter:
    """Use SOM for robust model fitting as an alternative to RANSAC.
    
    This class combines SOM with model fitting to achieve robust parameter
    estimation in the presence of outliers.
    """
    
    def __init__(self, model_func: Callable, som_config: Optional[SOMConfig] = None):
        """Initialize SOM-based model fitter.
        
        Args:
            model_func: Function that fits a model to data and returns parameters
                       Should have signature: func(data: np.ndarray) -> model_params
            som_config: Configuration for the SOM (optional)
        """
        self.model_func = model_func
        self.som_config = som_config
        self.som = None
        self.best_model = None
        self.inliers = None
    
    def fit(self, data: np.ndarray, outlier_percentile: float = 90,
            verbose: bool = False) -> 'SOMModelFitter':
        """Fit model using SOM for outlier rejection.
        
        Args:
            data: Input data of shape (n_samples, input_dim)
            outlier_percentile: Percentile threshold for outlier detection
            verbose: Whether to print progress
            
        Returns:
            Self for method chaining
        """
        # Create SOM config if not provided
        if self.som_config is None:
            self.som_config = SOMConfig(
                input_dim=data.shape[1],
                map_size=(10, 10),
                max_iterations=1000
            )
        
        # Train SOM
        self.som = KohonenSOM(self.som_config)
        self.som.fit(data, verbose=verbose)
        
        # Detect outliers
        self.inliers = self.som.detect_outliers(data, percentile=outlier_percentile)
        
        if verbose:
            n_inliers = np.sum(self.inliers)
            print(f"Detected {n_inliers}/{len(data)} inliers")
        
        # Fit model on inliers
        inlier_data = data[self.inliers]
        self.best_model = self.model_func(inlier_data)
        
        return self
    
    def get_inliers(self) -> np.ndarray:
        """Get boolean mask of inliers.
        
        Returns:
            Boolean array indicating inliers
        """
        return self.inliers
    
    def get_model(self):
        """Get fitted model parameters.
        
        Returns:
            Model parameters from model_func
        """
        return self.best_model
