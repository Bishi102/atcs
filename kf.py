"""
Kalman Filter:
"""

import numpy as np
from itertools import product

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
        # for graphing
        self.points = []
    
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
        # for graphing
        self.points.append(x)

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
    
    def getResults(self):
        """
        Given ordering of parameters and results are matching,
        the results can be simply extracted 1:1
        """
        return list(zip(self.points, self.mu))
    
    def getBest(self):
        """
        For returning best profit and parameters
        """
        bestIndex = np.argmax(self.mu)

        return self.points[bestIndex], self.mu[bestIndex]
    