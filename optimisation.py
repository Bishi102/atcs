"""
Optimisation Algorithm:

"""

from plots import Plots
from kf import KalmanFilter as Kf
from des import CoffeeShopSimulator as Simulator

import numpy as np
from itertools import product
from scipy.stats import qmc
from scipy.spatial import Delaunay

class Optimisation:

    def __init__(self, obj, bounds, 
                 nInit=10, 
                 nIter=10, 
                 nCand=10, 
                 beta=2.0,
                 epsilonStart=0.99,
                 epsilonEnd=0.1,
                 epsilonDecay=0.01):
        self.obj = obj
        self.bounds = bounds
        self.nInit = nInit
        self.nIter = nIter
        self.nCand = nCand
        self.beta = beta

        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecay = epsilonDecay

        self.kf = Kf()

    def initialTri(self):
        """
        Creates a Delaunay triangulation with corners being given parameter bounds
        """
        boundsArray = np.array(self.bounds)
        boundPairs = [(low, high) for low, high in boundsArray]

        corners = np.array(list(product(*boundPairs)))
        print("Corners:", corners)
        for corner in corners:
            self.kf.addPoint(self.obj(corner), corner)
        tri = Delaunay(corners, incremental=True, qhull_options='QJ')
        return tri
    
    def addPoint(self, tri, x, z):
        """
        z: observed variable (noisy)
        x: parameters of z
        """
        self.kf.addPoint(z, x)
        tri.add_points(np.array([x]), restart=False)


    def samplePoints(self, numOfPoints):
        """
        uses latin hypercube sampling to get initial points for evaluation and triangulation
        """
        dimension = self.bounds.shape[0]
        sampler = qmc.LatinHypercube(d=dimension)
        samples = sampler.random(n=numOfPoints)
        scaledSamples = qmc.scale(samples, self.bounds[:, 0], self.bounds[:, 1])
        return scaledSamples

    def interpolate(self, params, tri, mu, Sigma):
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
    
    def optimise(self):
        """
        Kalman-filter-guided optimization with Delaunay-based interpolation.

        obj: objective function
        Shows 2D plot only if parameter dimension is 2.
        """
        # initial triangulation of corners and formation of kalman filter
        tri = self.initialTri()
        
        # initial sampling and evaluation inside search space
        samples = self.samplePoints(self.nInit)
        for sample in samples:
            self.addPoint(tri, sample, self.obj(sample))

        # main optimisation loop (budget)
        for iteration in range(self.nIter):
            # epsilon greedy for exploration/exploitation of known/unknown point evaluations
            epsilon = max(self.epsilonEnd, self.epsilonStart * (self.epsilonDecay ** iteration))
            # getting best candidate(interpolated mean) of sampling
            candidates = self.samplePoints(self.nCand)
            acquisition = []
            for candidate in candidates:
                try:
                    mu, var = self.interpolate(candidate, tri, self.kf.mu, self.kf.Sigma)
                    ucb = mu + self.beta * np.sqrt(var)
                    acquisition.append((ucb, candidate))
                    #print(f"Candidate {candidate} is inside convex hull")
                except ValueError:
                    #print(f"Candidate {candidate} is outside convex hull")
                    continue
            print(len(acquisition))
            if not acquisition:
                print("No valid candidate points found inside the convex hull.")
                break
            acquisition.sort(reverse=True, key=lambda x: x[0])
            bestCandidate = acquisition[0][1]
            
            if np.random.rand() < epsilon:
                # explore: evaluate best candidate
                self.addPoint(tri, bestCandidate, self.obj(bestCandidate))
            else:
                # exploit: re-evaluate closest known point
                simplexIndex = tri.find_simplex(bestCandidate)
                if simplexIndex == -1:
                    raise ValueError("Point is outside the convex hull of known points.")
                simplexVertices = tri.simplices[simplexIndex]
                vertexPoints = np.array([self.kf.points[i] for i in simplexVertices])

                distances = np.linalg.norm(vertexPoints - bestCandidate, axis=1)
                closestIndex = simplexVertices[np.argmin(distances)]
                closestPoint = self.kf.points[closestIndex]
                self.kf.update(self.obj(closestPoint), closestIndex)
        # optimised resutl and parameters
        return self.kf.getBest()
