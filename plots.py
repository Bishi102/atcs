"""
Plotting functions
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import numpy as np

class Plots:
    def plot3DSurface(self, kf, tri):
        points = np.array(kf.points)
        mu = np.array(kf.mu)

        if points.shape[1] != 2:
            raise ValueError("3D plotting only supports 2D parameter space.")

        x = points[:, 0]
        y = points[:, 1]
        z = mu

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot triangulated surface
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', edgecolor='k', linewidth=0.5, alpha=0.8)

        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_zlabel('Predicted Mean')
        ax.set_title("3D Surface with Delaunay Triangulation")

        plt.tight_layout()
        plt.show()

    def plot(self, kf, tri, animate=True, save_path=None):
        points = np.array(kf.points)
        profits = np.array(kf.mu)

        if points.shape[1] != 2:
            raise ValueError("plot_results only supports 2D parameter spaces.")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        sc = ax.scatter(points[:, 0], points[:, 1], profits, c=profits, cmap='viridis', s=30)

        # Plot triangle faces
        triangles = []
        face_colors = []

        for simplex in tri.simplices:
            verts_2d = points[simplex]
            z_vals = profits[simplex]
            verts_3d = [(x, y, z) for (x, y), z in zip(verts_2d, z_vals)]
            triangles.append(verts_3d)
            face_colors.append(np.mean(z_vals))

        poly_collection = Poly3DCollection(triangles, cmap='viridis', edgecolor='gray', linewidth=0.3)
        poly_collection.set_array(np.array(face_colors))
        ax.add_collection3d(poly_collection)

        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")
        ax.set_zlabel("Estimated Profit")
        ax.set_title("3D Kalman Optimization Surface")

        # Adjust axis limits for better animation view
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        ax.set_zlim(profits.min(), profits.max())

        fig.colorbar(poly_collection, ax=ax, label="Mean Profit (z)")

        if animate:
            def rotate(angle):
                ax.view_init(elev=30, azim=angle)

            anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=100)

            if save_path:
                anim.save(save_path, writer='ffmpeg')
                print(f"Saved animation to {save_path}")
            else:
                plt.show()
        else:
            plt.show()
    def plot2DTri(self, tri, points):
        points = np.array(points)
        plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')
        plt.plot(points[:, 0], points[:, 1], 'o', color='red')
        plt.title("Initial Delaunay Triangulation")
        plt.xlabel("Parameter 1")
        plt.ylabel("Parameter 2")
        plt.axis("equal")
        plt.grid(True)
        plt.show()