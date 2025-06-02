import numpy as np
from scipy.stats import qmc
from scipy.spatial import Delaunay
from des import objective_function as obj

class KalmanFilter:
    def __init__(self):
        """
        mu: mu matrix (mean estimates)
        Sigma: covariance matrix
        Q = latent noise
        R: observation noise
        """
        self.mu = np.array([])
        self.Sigma = np.array([])
        self.Q = 1.0
        self.R = 0.04
    
    def addPoint(self, z, x):
        """
        z: observed variable (noisy)
        x: parameters of z
        """
        if self.mu.size == 0:
            self.mu = np.array([z])
            self.Sigma = np.array([1.0])
        else:
            self.mu = np.append(self.mu, z)
            numPoints = self.Sigma.shape[0]
            newRow = np.zeros((1, numPoints))
            self.Sigma = np.block([
                [self.Sigma, newRow.T],
                [newRow, np.array([1.0])]
            ])

    def update(self, z, index):
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
        simplified the below:
        sigmapost = sigmaprior - KHsigmaprior
        into the below:
        sigmapost = (I - KH)sigmaprior
        """
        I = np.eye(len(self.mu))
        self.Sigma = (I - K @ H) @ self.Sigma 
    
    
def sampleInitialPoints(numOfPoints, bounds):
    """
    uses latin hypercube sampling to get initial points for evaluation and triangulation
    """
    dimension = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=dimension)
    samples = sampler.random(n=numOfPoints)
    scaledSamples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    return scaledSamples

def interpolate(params, tri, mu, Sigma):
    """
    Uses barycentric coordinates inside Delaunay simplexes to interpolate mean
    Distance-based function used to estimate variance
    params: 1D np array parameter vector
    tri: Delaunay triangulation of known points
    mu: mean matrix
    Sigma: covariance matrix
    returns: interpolated mean and variance values
    """
    simplexIndex = tri.find_simplex(params)
    if simplexIndex == -1:
        raise ValueError("Point is outside the convex hull of known points.")
    vertices = tri.simplices[simplexIndex]
    T = tri.transform[simplexIndex]
    delta = params - T[-1]
    baryCoords = T[:-1] @ delta
    baryCoords = np.append(baryCoords, 1 - np.sum(baryCoords))

    mean = np.dot(baryCoords, mu[vertices])
    subSigma = Sigma[np.ix_(vertices, vertices)]
    variance = baryCoords @ subSigma @ baryCoords.T

    return mean, variance

def main(obj, bounds, nInit, nIter, nCand, beta):
      """
      
      """
      # intial parameter sampling
      X = sampleInitialPoints(nInit, bounds)
      Y = np.array([obj(x) for x in X])
      print("Initial X:", X)
      print("Initial Y:", Y)
      # initialise kalman filter and add points
      kf = KalmanFilter()
      for i in range(nInit):
          kf.addPoint(Y[i], X[i])
      
      # triangulate evaluated points (params, obj(params))
      points = np.hstack([X, Y.reshape(-1, 1)])
      tri = Delaunay(points)

      for iteration in range(nIter):
          # sampling of candidates
          candidates = sampleInitialPoints(nCand, bounds)
          acquisition = []
          for c in candidates:
              try:
                  mu, var = interpolate(c, tri, kf.mu, kf.Sigma)
                  print(mu)
                  print(var)
                  ucb = mu + beta * np.sqrt(var)
                  print(ucb)
                  acquisition.append((ucb, c))
              except ValueError:
                  print(f"Candidate {c} is outside convex hull")
                  continue
          if not acquisition:
              print("No valid candidate points found inside the convex hull.")
              break
          # get best candidate based on acquisition function
          acquisition.sort(reverse=True, key=lambda x: x[0])
          bestCandidate = acquisition[0][1]

          # evaluate at best candidate params
          newY = obj(bestCandidate)

          # update kalman filter and add new point to known points and update triangulation
          kf.addPoint(newY, bestCandidate)
          X = np.vstack([X, bestCandidate])
          Y = np.append(Y, newY)
          points = np.hstack([X, Y.reshape(-1, 1)])
          tri = Delaunay(points)

          print("iteration " + iteration + " complete")
          bestIndex = np.argmax(Y)
      return X[bestIndex], Y[bestIndex]

if __name__ == "__main__":
    bounds = [
    (2.5, 8.0),    # coffee_price
    (1, 5)        # num_baristas
]
    bounds = np.array(bounds)
    params, profit = main(obj, bounds, 10, 10, 100, 2)
    print(params)
    print(profit)
          