from tqdm import tqdm
import math
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import numpy as np
import os

class OBJ:
	def __init__(self, foldername, filename, swapyz=False):
		"""Loads a Wavefront OBJ file. """
		self.vertices = []
		self.normals = []
		self.texcoords = []
		self.faces = []

		material = None
		print("Loading Geometry...")
		for line in tqdm(open(foldername + "/" + filename, "r")):
			if line.startswith('#'): continue
			values = line.split()
			if not values: continue
			if values[0] == 'v':
				v = list(map(float, values[1:4]))
				if swapyz:
					v = v[0], v[2], v[1]
				self.vertices.append(v)
			elif values[0] == 'vn':
				v = list(map(float, values[1:4]))
				if swapyz:
					v = v[0], v[2], v[1]
				self.normals.append(v)
			elif values[0] == 'vt':
				self.texcoords.append(list(map(float, values[1:3])))
			elif values[0] == 'f':
				face = []
				texcoords = []
				norms = []
				for v in values[1:]:
					w = v.split('/')
					face.append(int(w[0]))
					if len(w) >= 2 and len(w[1]) > 0:
						texcoords.append(int(w[1]))
					else:
						texcoords.append(0)
					if len(w) >= 3 and len(w[2]) > 0:
						norms.append(int(w[2]))
					else:
						norms.append(0)
				self.faces.append((face, norms, texcoords, material))
		
		print("Orienting Geometry...")
		self.orient()

	def rotate(self, angle):
		for i in range(len(self.vertices)):
			v = self.vertices[i]
			self.vertices[i] = v[0] * math.cos(angle) - v[1] * math.sin(angle), v[0] * math.sin(angle) + v[1] * math.cos(angle), v[2]
			vn = self.normals[i]
			self.normals[i] = vn[0] * math.cos(angle) - vn[1] * math.sin(angle), vn[0] * math.sin(angle) + vn[1] * math.cos(angle), vn[2]

	def scale(self, s):
		for i in range(len(self.vertices)):
			v = self.vertices[i]
			self.vertices[i] = v[0] * s, v[1] * s, v[2] * s
	
	def center(self):
		bbox = self.BBox()
		for i in range(len(self.vertices)):
			v = self.vertices[i]
			self.vertices[i] = v[0] - (bbox[1] + bbox[0]) / 2, v[1] - (bbox[3] + bbox[2]) / 2, v[2] - bbox[4]
	
	def orient(self, debug = False):
		vertices = [list(v) for v in self.vertices]
		sample_points = np.random.choice(len(vertices), int(len(vertices) / 10)).tolist()
		sample_vertices = np.array(vertices)[sample_points]

		norm = [1, 0]
		theta = math.atan(norm[1] / norm[0])

		if (debug):
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')

			markers = ['o', 'v', '^', '<', '>', '+']
			for i, points in enumerate(plane_points):
				ax.scatter(
					plane_points[i][:,0],
					plane_points[i][:,1],
					plane_points[i][:,2],
					color = (0, 0, 0, 0.5),
					marker = markers[i % 6]
				)
			ax.set_xlabel('X Label')
			ax.set_ylabel('Y Label')
			ax.set_zlabel('Z Label')
			plt.show()

		print("Rotating...")
		self.rotate(-theta)
		print("Centering...")
		self.center()
		self.scale(100)

	def BBox(self):
		left = min(self.vertices, key = lambda x: x[0])
		right = max(self.vertices, key = lambda x: x[0])
		front = min(self.vertices, key = lambda x: x[1])
		back = max(self.vertices, key = lambda x: x[1])
		top = min(self.vertices, key = lambda x: x[2])
		bottom = max(self.vertices, key = lambda x: x[2])
		return left[0], right[0], front[1], back[1], top[2], bottom[2]

	def height(self):
		return max(self.vertices, key = lambda x: x[2])[2] - min(self.vertices, key = lambda x: x[2])[2]

	def export(self, filename):
		with open(filename, 'w') as file:
			for v in self.vertices:
				file.write(f'v {v[0]} {v[1]} {v[2]}\n')
			for face in self.faces:
				vertices, _, _, _ = face
				# OBJ format uses 1-based indexing
				face_indices = [str(v + 1) for v in vertices]
				file.write(f'f {" ".join(face_indices)}\n')

model = OBJ("Scan", "test2.obj", swapyz=True)
normals = model.normals
normals_nonvertical = []

print(len(normals))

for v in normals:
	if (v[2] < 0.1 and v[2] > -0.1):
		normals_nonvertical.append(v)

print(len(normals_nonvertical))

sample_indices = np.random.choice(len(normals_nonvertical), int(len(normals_nonvertical) / 100)).tolist()
sample_normals = np.array(normals_nonvertical)[sample_indices]

dbscan = DBSCAN(eps=0.05, min_samples=10)
clusters = dbscan.fit_predict(sample_normals)
print(clusters)

unique_clusters, counts = np.unique(clusters[clusters >= 0], return_counts=True)
print(counts)
max_cluster = unique_clusters[np.argmax(counts)]
cluster_normals = sample_normals[clusters == max_cluster]
cluster_mean = np.mean(cluster_normals, axis=0)
print(cluster_mean)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = ['blue', 'green', 'red', 'purple', 'orange', (0, 0, 0, 0.1)]

for i, p in enumerate(sample_normals):
    ax.scatter(
        p[0],
        p[1],
        p[2],
        color = colors[clusters[i] % len(colors)],
        marker = 'o'
    )
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()