"""

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plots:
    def plot_surface(results):
        """
        results: list of (params, mean)
        params: 2D vector
        mean: scalar
        """
        x = [p[0] for p, _ in results]
        y = [p[1] for p, _ in results]
        z = [m for _, m in results]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=z, cmap='viridis')
        ax.set_xlabel("Param 1")
        ax.set_ylabel("Param 2")
        ax.set_zlabel("Estimated Mean (mu)")
        ax.set_title("Kalman Filter Estimated Surface")
        plt.show()