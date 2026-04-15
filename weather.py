import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Synthetic Weather-like Data (Non-linear)
np.random.seed(42)
X = np.linspace(0, 24, 100)  # 24 hours
# Simulating temperature fluctuation with a sine wave + noise
y = 25 + 10 * np.sin(X * np.pi / 12) + np.random.normal(0, 1.5, 100)

# Reshape X for matrix operations
X_mat = X.reshape(-1, 1)

def get_weights(query_point, X, tau):
    """
    Calculates the Gaussian Kernel weights for a specific query point.
    Formula: w = exp(-(x - x_i)^2 / (2 * tau^2))
    """
    m = X.shape[0]
    weights = np.eye(m) # Initialize identity matrix
    
    for i in range(m):
        diff = query_point - X[i]
        weights[i, i] = np.exp(np.dot(diff, diff.T) / (-2.0 * tau**2))
    
    return weights

def predict(X, y, query_point, tau):
    """
    Solves the Weighted Normal Equation: theta = (X.T * W * X)^-1 * X.T * W * y
    """
    # Add bias term (1) to the input
    m = X.shape[0]
    X_bias = np.append(np.ones((m, 1)), X, axis=1)
    
    query_bias = np.array([1, query_point])
    
    # Get Weights for the current query point
    W = get_weights(query_bias, X_bias, tau)
    
    # Calculate Theta using the Normal Equation
    # (X.T * W * X)
    XTWX = X_bias.T @ W @ X_bias
    # (X.T * W * y)
    XTWy = X_bias.T @ W @ y
    
    # Inverse of XTWX and multiply with XTWy
    try:
        theta = np.linalg.pinv(XTWX) @ XTWy
    except np.linalg.LinAlgError:
        return None
        
    # Final Prediction: y = theta * query
    return query_bias @ theta

# 2. Testing the model
tau = 0.5  # Bandwidth parameter (Tuning this changes the smoothness)
query_x = 14.5 # Predicting for 2:30 PM
prediction = predict(X_mat, y, query_x, tau)

print(f"Prediction for hour {query_x}: {prediction:.2f}°C")

# 3. Visualization
X_test = np.linspace(0, 24, 50)
predictions = [predict(X_mat, y, x_point, tau) for x_point in X_test]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='lightblue', label='Historical Data Points')
plt.plot(X_test, predictions, color='red', linewidth=2, label=f'LWR Curve (tau={tau})')
plt.title('Non-Parametric Weather Prediction (LWR)')
plt.xlabel('Time of Day (Hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()