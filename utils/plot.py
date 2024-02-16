import numpy as np
import matplotlib.pyplot as plt

def plotMesh2D(vertices: np.ndarray, triangles: list):
    for triangle in triangles:
        print(triangle)
        plt.plot([vertices[triangle[0]][0], vertices[triangle[1]][0]], [vertices[triangle[0]][1], vertices[triangle[1]][1]], marker = 'o')
        plt.plot([vertices[triangle[0]][0], vertices[triangle[2]][0]], [vertices[triangle[0]][1], vertices[triangle[2]][1]], marker = 'o')
        plt.plot([vertices[triangle[1]][0], vertices[triangle[2]][0]], [vertices[triangle[1]][1], vertices[triangle[2]][1]], marker = 'o')
    plt.show()