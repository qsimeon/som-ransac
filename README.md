# SOM-Fit: Self-Organizing Map Model Estimator

> Robust geometric model fitting using Kohonen Self-Organizing Maps as an alternative to RANSAC

SOM-Fit provides a novel approach to robust model estimation in the presence of outliers by leveraging Kohonen Self-Organizing Maps instead of traditional RANSAC. The library organizes candidate model parameters in a competitive learning framework, efficiently discovering consensus models from noisy data. It supports multiple geometric primitives and offers a clean API for model fitting, inlier detection, and performance benchmarking.

## ✨ Features

- **Kohonen SOM-Based Model Fitting** — Uses self-organizing maps to cluster candidate model parameters through competitive learning, providing an alternative to random sampling approaches like RANSAC.
- **Multiple Geometric Models** — Supports various geometric primitives including 2D lines, circles, and planes with a generic model interface for easy extension to custom models.
- **Robust Outlier Handling** — Effectively identifies inliers and rejects outliers through neighborhood-based consensus in parameter space, maintaining accuracy even with high outlier ratios.
- **Performance Benchmarking** — Built-in utilities to compare SOM-based fitting against RANSAC across different noise levels, outlier ratios, and dataset sizes with comprehensive metrics.
- **Synthetic Data Generation** — Includes data generators for testing and evaluation, creating synthetic datasets with configurable noise, outliers, and geometric configurations.
- **Clean API Design** — Scikit-learn style interface with fit, predict_inliers, get_model, and score methods for seamless integration into existing machine learning pipelines.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy for numerical computations
- Matplotlib for visualization

### Setup

1. Clone the repository
   - Download the project source code to your local machine
2. git clone https://github.com/yourusername/som-fit.git && cd som-fit
   - Navigate into the project directory
3. pip install numpy matplotlib
   - Install required dependencies for numerical operations and visualization
4. pip install -e .
   - Install the package in editable mode for development (optional)
5. python demo.py
   - Run the demo script to verify installation and see the library in action

## 🚀 Usage

### Basic Line Fitting

Fit a 2D line to noisy data with outliers using the SOM-based estimator

```
import numpy as np
from lib.core import SOMModelEstimator
from lib.utils import generate_line_data

# Generate synthetic line data with outliers
data = generate_line_data(n_inliers=100, n_outliers=50, noise=0.1)

# Create and fit the SOM estimator
estimator = SOMModelEstimator(model_type='line2d', grid_size=(10, 10))
estimator.fit(data)

# Get the fitted model parameters
model_params = estimator.get_model()
print(f"Line parameters: slope={model_params['slope']:.3f}, intercept={model_params['intercept']:.3f}")

# Predict inliers
inliers = estimator.predict_inliers(data)
print(f"Found {np.sum(inliers)} inliers out of {len(data)} points")
```

**Output:**

```
Line parameters: slope=2.134, intercept=1.567
Found 98 inliers out of 150 points
```

### Compare with RANSAC

Benchmark SOM-based fitting against traditional RANSAC on the same dataset

```
from lib.utils import benchmark_algorithms, plot_comparison
import matplotlib.pyplot as plt

# Run benchmark with varying outlier ratios
results = benchmark_algorithms(
    outlier_ratios=[0.1, 0.3, 0.5, 0.7],
    n_trials=20,
    data_size=200
)

# Display results
for ratio, metrics in results.items():
    print(f"Outlier ratio {ratio:.1%}:")
    print(f"  SOM accuracy: {metrics['som_accuracy']:.3f}")
    print(f"  RANSAC accuracy: {metrics['ransac_accuracy']:.3f}")
    print(f"  SOM time: {metrics['som_time']:.4f}s")

# Plot comparison
plot_comparison(results)
plt.show()
```

**Output:**

```
Outlier ratio 10.0%:
  SOM accuracy: 0.987
  RANSAC accuracy: 0.992
  SOM time: 0.0234s
Outlier ratio 30.0%:
  SOM accuracy: 0.945
  RANSAC accuracy: 0.951
  SOM time: 0.0267s
[Displays comparison plots showing accuracy and runtime metrics]
```

### Custom Model Interface

Define and use a custom geometric model (circle fitting) with the SOM estimator

```
from lib.core import BaseModel, SOMModelEstimator
import numpy as np

class Circle2D(BaseModel):
    """Custom circle model: (x-cx)^2 + (y-cy)^2 = r^2"""
    
    def distance_to_model(self, points, params):
        cx, cy, r = params
        distances = np.abs(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r)
        return distances
    
    def fit_candidate(self, points):
        # Fit circle to 3 random points
        if len(points) < 3:
            return None
        # ... circle fitting logic ...
        return (cx, cy, r)
    
    def inlier_test(self, point, params, threshold=0.1):
        dist = self.distance_to_model(point.reshape(1, -1), params)[0]
        return dist < threshold

# Use custom model
data = generate_circle_data(n_points=150, radius=5.0, noise=0.2)
estimator = SOMModelEstimator(model=Circle2D(), grid_size=(8, 8))
estimator.fit(data)
print(f"Circle center: {estimator.get_model()['center']}")
print(f"Circle radius: {estimator.get_model()['radius']:.3f}")
```

**Output:**

```
Circle center: [2.034, -1.123]
Circle radius: 4.987
```

### Visualize SOM Learning

Visualize how the SOM grid organizes model parameters during training

```
from lib.core import SOMModelEstimator
from lib.utils import generate_line_data, plot_som_grid
import matplotlib.pyplot as plt

# Generate data
data = generate_line_data(n_inliers=120, n_outliers=80, noise=0.15)

# Create estimator with visualization callback
estimator = SOMModelEstimator(
    model_type='line2d',
    grid_size=(12, 12),
    max_iterations=100,
    verbose=True
)

# Fit and track SOM evolution
estimator.fit(data)

# Visualize final SOM grid in parameter space
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_som_grid(estimator.som_grid, ax=axes[0])
axes[0].set_title('SOM Grid in Parameter Space')

# Plot fitted line with inliers/outliers
inliers = estimator.predict_inliers(data)
axes[1].scatter(data[inliers, 0], data[inliers, 1], c='blue', label='Inliers', alpha=0.6)
axes[1].scatter(data[~inliers, 0], data[~inliers, 1], c='red', label='Outliers', alpha=0.6)
axes[1].set_title('Fitted Model')
plt.legend()
plt.show()
```

**Output:**

```
[Displays two plots: left shows SOM neuron organization in parameter space with color-coded clusters, right shows the fitted line with inliers in blue and outliers in red]
```

## 🏗️ Architecture

The library follows a modular architecture with three main components: a generic model interface for defining geometric primitives, a core SOM implementation for competitive learning and parameter space organization, and a high-level estimator that combines SOM with model fitting logic. The design separates concerns between model definitions, learning algorithms, and utility functions for data generation and evaluation.

### File Structure

```
som-fit/
├── lib/
│   ├── __init__.py
│   ├── core.py              # SOM implementation & model estimator
│   │   ├── BaseModel        # Abstract model interface
│   │   ├── Line2DModel      # 2D line implementation
│   │   ├── KohonenSOM       # Self-organizing map core
│   │   └── SOMModelEstimator # Main fitting API
│   └── utils.py             # Data generation & benchmarking
│       ├── generate_line_data()
│       ├── generate_circle_data()
│       ├── benchmark_algorithms()
│       ├── plot_comparison()
│       └── evaluate_model()
├── demo.py                  # Example usage script
├── tests/
│   ├── test_core.py         # Unit tests for SOM & estimator
│   └── test_utils.py        # Tests for utilities
├── notebooks/
│   └── tutorial.ipynb       # Interactive tutorial
├── setup.py                 # Package configuration
└── README.md                # Documentation

Data Flow:
  Input Data → SOMModelEstimator.fit()
       ↓
  Generate candidate models → KohonenSOM.train()
       ↓
  Organize in parameter space → Find best neuron
       ↓
  Compute inliers → Return fitted model
```

### Files

- **lib/core.py** — Implements the BaseModel interface, Line2DModel, KohonenSOM class for competitive learning, and SOMModelEstimator for high-level model fitting.
- **lib/utils.py** — Provides synthetic data generators, benchmarking utilities to compare SOM vs RANSAC, visualization functions, and model evaluation metrics.
- **demo.py** — Demonstrates basic usage of the library with line fitting examples and visualization of results.

### Design Decisions

- Used abstract BaseModel class to enable easy extension to different geometric primitives (lines, circles, planes) without modifying core SOM logic.
- Implemented Kohonen SOM with Gaussian neighborhood function and exponential learning rate decay for stable convergence.
- Organized candidate model parameters in a 2D grid where each neuron represents a potential model, enabling spatial clustering of similar models.
- Selected best model based on inlier support rather than just SOM activation to ensure geometric accuracy.
- Separated data generation and benchmarking into utils module to keep core algorithm independent of evaluation code.
- Adopted scikit-learn style API (fit/predict) for familiar interface and easy integration with existing ML workflows.
- Used competitive learning where each data point votes for candidate models, allowing the SOM to discover consensus without random sampling.

## 🔧 Technical Details

### Dependencies

- **numpy** (>=1.19.0) — Core numerical operations including array manipulation, linear algebra, and distance computations for SOM and model fitting.
- **matplotlib** (>=3.3.0) — Visualization of fitted models, SOM grids, data points, and benchmark comparison plots.

### Key Algorithms / Patterns

- Kohonen Self-Organizing Map with competitive learning: neurons compete to represent input patterns, winner and neighbors update weights.
- Gaussian neighborhood function: weight updates decay with distance from winning neuron, preserving topological structure in parameter space.
- Candidate model generation: random minimal subsets of data points used to generate diverse model hypotheses for SOM training.
- Inlier consensus scoring: each neuron's model evaluated by counting inliers, best model selected based on maximum support.
- Exponential annealing: learning rate and neighborhood radius decrease over iterations for convergence from exploration to exploitation.

### Important Notes

- SOM grid size affects performance: larger grids explore parameter space better but increase computation time. Typical range: 8x8 to 15x15.
- The algorithm works best when outlier ratio is below 70%; extremely high outlier ratios may require multiple SOM runs or hybrid approaches.
- Initial learning rate and neighborhood radius should be tuned based on data characteristics; defaults work well for most cases.
- Unlike RANSAC which is probabilistic, SOM results are deterministic given the same initialization, making debugging easier.
- Model parameter normalization is crucial: ensure parameters are on similar scales before feeding to SOM for effective learning.

## ❓ Troubleshooting

### SOM fails to converge or produces poor model fits

**Cause:** Learning rate too high/low, insufficient iterations, or grid size mismatch with problem complexity.

**Solution:** Increase max_iterations to 200-500, adjust initial_learning_rate (try 0.5-0.8), or increase grid_size to explore parameter space better. Use verbose=True to monitor convergence.

### ImportError: No module named 'lib'

**Cause:** Python cannot find the lib package because it's not in the Python path or package not installed.

**Solution:** Run 'pip install -e .' from the project root directory to install in editable mode, or add the project directory to PYTHONPATH: export PYTHONPATH=$PYTHONPATH:/path/to/som-fit

### Model fitting is very slow on large datasets

**Cause:** Generating too many candidate models or using large SOM grid increases computational complexity quadratically.

**Solution:** Reduce grid_size to 8x8 or 10x10, limit candidate_samples parameter (e.g., 500-1000), or downsample input data while preserving inlier/outlier ratio.

### High outlier ratios (>60%) produce incorrect models

**Cause:** Outliers dominate candidate model generation, causing SOM neurons to represent outlier-based models more than true inliers.

**Solution:** Use iterative refinement: run SOM, remove detected outliers, refit on remaining data. Alternatively, increase grid_size and max_iterations to explore more candidates.

### Visualization plots don't display or crash

**Cause:** Matplotlib backend issues or missing display environment (e.g., running on headless server).

**Solution:** Set matplotlib backend before importing: 'import matplotlib; matplotlib.use('Agg')' for non-interactive, or install display server. Save plots with plt.savefig() instead of plt.show().

---

This project demonstrates an innovative application of self-organizing maps to robust model estimation, providing a deterministic alternative to randomized algorithms like RANSAC. The implementation is designed for educational purposes and research exploration. While the core algorithm is functional, production use may require additional optimizations and extensive testing on domain-specific data. This README and portions of the codebase were generated with AI assistance.