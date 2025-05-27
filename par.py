import numpy as np
from scipy.stats import qmc
from scipy.spatial import Delaunay

class KalmanPoint:
    def __init__(self, x, z, sigma=1.0, r=1.0):
        """
        x: prior mean
        z: observed measurement
        sigma: prior variance
        r: estimated variance
        """
        self.x = x
        self.sigma = sigma
        self.z = z
        self.r = r
        self.numOfUpdates = 0
    
    def updateKalman(self, z):
        """
        Perform scalar Kalman filter update for a single point.

        Returns: updated_mean, updated_variance
        """
        K = self.sigma / (self.sigma + self.r)
        postMean = self.x + K * (z - self.x)
        postVariance = (1 - K) * self.sigma

        return postMean, postVariance
        
def sampleInitialPoints(numOfPoints, bounds):
    """
    uses latin hypercube sampling to get initial points for evaluation and triangulation
    """
    dimension = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=dimension)
    samples = sampler.random(n=numOfPoints)
    scaledSamples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    return scaledSamples

def simulate_profit(x):
    """
    Black-box function: returns a noisy scalar profit.
    """
    true_value = -np.sum((x - 0.5)**2) 
    noise = np.random.normal(0, 0.2)
    return true_value + noise

# Setup
bounds = np.array([[0.0, 1.0]] * 3)  # for 3D example
num_points = 10
noise_variance = 0.04  # known noise variance (0.2^2)

# Step 1: Sample initial points
X_init = sampleInitialPoints(num_points, bounds)

# Step 2: Evaluate and apply Kalman filter
points = []
print("Initial evaluation:")
for i, x in enumerate(X_init):
    z = simulate_profit(x)
    point = KalmanPoint(x, z, sigma=1.0, r=noise_variance)
    points.append(point)
    print(f"Point {i}: x = {x}")
    print(f"  Initial observation (z): {z:.4f}")
    print(f"  Initial mean: {point.z:.4f}")
    print(f"  Initial variance: {point.sigma:.4f}\n")

# Optional: Re-evaluate a specific point and update
i = 0  # index of the point to re-evaluate
print("Re-evaluating point 0:")
new_observation = simulate_profit(points[i].x)
points[i].updateKalman(new_observation)
print(f"Point {i}: x = {x}")
print(f"  Initial observation (z): {z:.4f}")
print(f"  Initial mean: {point.z:.4f}")
print(f"  Initial variance: {point.sigma:.4f}\n")