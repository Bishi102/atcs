import numpy as np
from scipy.stats import qmc
from scipy.spatial import Delaunay

class KalmanFilter:
    def __init__(self):
        """
        mu: mu matrix (mean estimates)
        sigma: covariance matrix
        Q = noise in x (latent)
        R: observation noise
        """
        self.mu = np.array([])
        self.sigma = np.array([])
        self.Q = 1.0
        self.R = 0.04
    
    def addPoint(self, z, index=None):
        """
        z: observed variable (noisy)
        """
        if index is None:
            # must be a new point
            if self.mu.size() == 0:
                # preventing possible bugs from appending to empty np array
                self.mu = np.array([z])
                self.sigma = np.array([1.0])
            else:
                self.mu = np.append(self.mu, z)
                # expanding matrix by 1 row and col, setting prior variance to 1.0 as default
                numPoints = self.sigma.shape[0]
                newRow = np.zeros((1, numPoints))
                self.sigma = np.block([
                    [self.sigma, newRow.T],
                    [newRow, np.array([1.0])]
                ])
        else:
            H = np.zeros((1, self.mu.shape[0]))
            H[0, index] = 1.0

            # kalman gain (bayesian update term)
            S = H @ self.Sigma @ H.T + self.R
            K = self.Sigma @ H.T @ np.linalg.inv(S)

            # posterior mean
            y = z - (H @ self.mu)
            self.mu = self.mu + (K @ y).flatten()

            # posterior covariance
            """
            i simplify the below:
            sigmapost = sigmaprior - KHsigmaprior
            into the below:
            sigmapost = (I - KH)sigmaprior
            """
            I = np.eye(len(self.mu))
            self.Sigma = (I - K @ H) @ self.Sigma
    def getState(self):
        return self.mu, self.sigma
        
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