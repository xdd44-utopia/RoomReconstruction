import numpy as np
from utils.mesh import *
from utils.plot import plotMesh2D, plotLine2D
from utils.object import OBJ
import matplotlib.pyplot as plt

# vertices = np.asarray([
#     [-1.194, -7.084, 0.0],
#     [-6.131, -0.213, 0.0],
#     [-7.482, 4.957, 0.0],
#     [-3.196, 7.170, 0.0],
#     [-1.333, 3.164, 0.0],
#     [2.743, 6.145, 0.0],
#     [5.141, 1.976, 0.0],
#     [0.903, 2.139, 0.0],
#     [6.562, -1.937, 0.0],
#     [10.545, 7.286, 0.0],
#     [13.619, 3.024, 0.0],
#     [5.095, -6.478, 0.0],
#     [1.694, -4.732, 0.0],
#     [-2.498, -0.376, 0.0]
# ])
# boundary = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# result = triangulizePolygon(vertices, boundary)
# plotMesh2D(vertices, result)

# boundary = extractConvexBoundary(vertices)

model = OBJ("test.obj", swapyz=True)
boundary = model.getBoundary()

for i in range(len(boundary) - 1):
    plotLine2D(vertices[i], vertices[i + 1])

# result = triangulizePolygon(model.vertices, boundary)
# print(result)
# plotMesh2D(vertices, result)

plt.show()