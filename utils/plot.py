import numpy as np
import matplotlib.pyplot as plt

def plotLine2D(v1, v2, color = 'blue', marker ='o'):
    plt.plot([v1[0], v2[0]], [v1[1], v2[1]], color = color, marker = marker)

def plotBoundary2D(vertices, boundary, color = 'blue', marker = 'o'):
    for i in range(len(boundary)):
        plotLine2D(vertices[boundary[i]], vertices[boundary[(i + 1) % len(boundary)]], color, marker)

def plotMesh2D(vertices: np.ndarray, triangles: list):
    for triangle in triangles:
        plotLine2D(vertices[triangle[0]], vertices[triangle[1]])
        plotLine2D(vertices[triangle[0]], vertices[triangle[2]])
        plotLine2D(vertices[triangle[1]], vertices[triangle[2]])