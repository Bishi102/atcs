import numpy as np
from scipy.stats import qmc
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    Kalman-filter-guided optimization with Delaunay-based interpolation.
    Shows 2D plot only if parameter dimension is 2.
    """
    dim = bounds.shape[0]

    # initial parameter sampling
    X = sampleInitialPoints(nInit, bounds)
    Y = np.array([obj(x) for x in X])
    print("Initial X:", X)
    print("Initial Y:", Y)

    # initialise Kalman filter and add points
    kf = KalmanFilter()
    for i in range(nInit):
        kf.addPoint(Y[i], X[i])

    # initial Delaunay triangulation
    tri = Delaunay(X, furthest_site=False)

    for iteration in range(nIter):
        candidates = sampleInitialPoints(nCand, bounds)
        acquisition = []
        for c in candidates:
            try:
                mu, var = interpolate(c, tri, kf.mu, kf.Sigma)
                ucb = mu + beta * np.sqrt(var)
                acquisition.append((ucb, c))
            except ValueError:
                print(f"Candidate {c} is outside convex hull")
                continue
        if not acquisition:
            print("No valid candidate points found inside the convex hull.")
            break

        # pick best candidate
        acquisition.sort(reverse=True, key=lambda x: x[0])
        bestCandidate = acquisition[0][1]

        # evaluate and update
        newY = obj(bestCandidate)
        kf.addPoint(newY, bestCandidate)
        X = np.vstack([X, bestCandidate])
        Y = np.append(Y, newY)
        tri = Delaunay(X)

        print("Iteration", iteration, "complete")

    # best found
    bestIndex = np.argmax(Y)
    bestParams = X[bestIndex]
    bestProfit = Y[bestIndex]

    # --- Plot only if input is 2D ---
    if dim == 2:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter all evaluated points
        ax.scatter(X[:, 0], X[:, 1], Y, c=Y, cmap='viridis', s=40, label="Evaluated points")
        
        # Highlight best point
        ax.scatter(bestParams[0], bestParams[1], bestProfit, color='red', s=100, edgecolor='black', label="Best point")

            # Plot Delaunay triangulation as 3D surface
        if tri.points.shape[1] == 2:
            for simplex in tri.simplices:
                verts = [(X[i][0], X[i][1], Y[i]) for i in simplex]
                tri_poly = Poly3DCollection([verts], alpha=0.3)
                tri_poly.set_color('lightblue')
                tri_poly.set_edgecolor('gray')
                ax.add_collection3d(tri_poly)


        ax.set_xlabel("Coffee Price")
        ax.set_ylabel("Number of Baristas")
        ax.set_zlabel("Profit")
        ax.set_title("3D Optimization Surface (params vs f(params))")
        ax.legend()
        plt.tight_layout()
        plt.show()


    return bestParams, bestProfit


if __name__ == "__main__":
    bounds = [
        (2.5, 8.0),    # coffee_price
        (1, 5)         # num_baristas
    ]
    bounds = np.array(bounds)
    params, profit = main(obj, bounds, 10, 10, 100, 2)
    print(params)
    print(profit)
