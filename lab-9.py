import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Generate dataset
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)  # Non-linear function with noise

def weights(x_query, X, tau):
    """Compute weights for locally weighted regression."""
    return np.diag(np.exp(-((X - x_query) ** 2) / (2 * tau ** 2)))

def locally_weighted_regression(x_query, X, Y, tau):
    """Perform Locally Weighted Regression for a given query point."""
    X_bias = np.vstack((np.ones_like(X), X)).T  # Add bias term
    W = weights(x_query, X, tau)
    theta = inv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ Y  # Compute theta
    return np.array([1, x_query]) @ theta  # Predict value

# Fit LWR for multiple points
x_fit = np.linspace(0, 10, 100)
y_fit = np.array([locally_weighted_regression(xq, x, y, tau=0.5) for xq in x_fit])

# Visualization
plt.scatter(x, y, label='Data', color='blue', alpha=0.5)
plt.plot(x_fit, y_fit, label='LWR Fit (τ=0.5)', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Locally Weighted Regression')
plt.show()

