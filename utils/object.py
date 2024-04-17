import pygame
from OpenGL.GL import *
from tqdm import tqdm
import math
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import numpy as np
import os

from utils.mesh import extractConvexBoundary

def MTL(foldername, filename):
	contents = {}
	mtl = None
	for line in open(foldername + "/" + filename, "r"):
		if line.startswith('#'): continue
		values = line.split()
		if not values: continue
		if values[0] == 'newmtl':
			mtl = contents[values[1]] = {}
		elif mtl is None:
			raise ValueError
		elif values[0] == 'map_Kd':
			# load the texture referred to by this declaration
			mtl[values[0]] = values[1]
			surf = pygame.image.load(foldername + "/" + mtl['map_Kd'])
			image = pygame.image.tostring(surf, 'RGBA', 1)
			ix, iy = surf.get_rect().size
			texid = mtl['texture_Kd'] = glGenTextures(1)
			glBindTexture(GL_TEXTURE_2D, texid)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
				GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
				GL_LINEAR)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
				GL_UNSIGNED_BYTE, image)
		else:
			mtl[values[0]] = map(float, values[1:])
	return contents

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
			elif values[0] in ('usemtl', 'usemat'):
				material = values[1]
			elif values[0] == 'mtllib':
				self.mtl = MTL(foldername, values[1])
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

		self.gl_list = glGenLists(1)
		glNewList(self.gl_list, GL_COMPILE)
		glEnable(GL_TEXTURE_2D)
		glFrontFace(GL_CCW)

		print("Loading Texture...")
		for face in tqdm(self.faces):
			vertices, normals, texture_coords, material = face

			mtl = self.mtl[material]
			if 'texture_Kd' in mtl:
				# use diffuse texmap
				glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
			else:
				# just use diffuse colour
				glColor(*mtl['Kd'])

			glBegin(GL_POLYGON)
			for i in range(len(vertices)):
				if normals[i] > 0:
					glNormal3fv(self.normals[normals[i] - 1])
				if texture_coords[i] > 0:
					glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
				glVertex3fv(self.vertices[vertices[i] - 1])
			glEnd()
		glDisable(GL_TEXTURE_2D)
		glEndList()

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
		normals = []
		for v in self.normals:
			if (v[2] < 0.05 and v[2] > -0.05):
				normals.append(v)
		sample_indices = np.random.choice(len(normals), int(len(normals) / 10)).tolist()
		sample_normals = np.array(normals)[sample_indices]
		dbscan = DBSCAN(eps=0.04, min_samples=20)
		clusters = dbscan.fit_predict(sample_normals)
		unique_clusters, counts = np.unique(clusters[clusters >= 0], return_counts=True)
		max_cluster = unique_clusters[np.argmax(counts)]
		cluster_normals = sample_normals[clusters == max_cluster]
		cluster_mean = np.mean(cluster_normals, axis=0)
		theta = math.atan(cluster_mean[1] / cluster_mean[0])

		if (debug):
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			
			# ax.scatter(
			# 	np.asarray(vertices)[:,0],
			# 	np.asarray(vertices)[:,1],
			# 	np.asarray(vertices)[:,2],
			# 	color = (0, 0, 0.5, 0.5),
			# 	marker = '.'
			# )

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

class NakedOBJ:
	def __init__(self, filename = "", vertices = [], faces = [], texcoords = [], texidc = [], bottomTriangleCount = 0):
		"""Loads a Wavefront OBJ file. """
		self.vertices = vertices
		self.faces = faces
		self.texcoords = texcoords
		self.texidc = texidc
		self.triCount = bottomTriangleCount
	
		if (len(filename) > 0):
			print("Loading Geometry...")
			for line in tqdm(open(filename, "r")):
				if line.startswith('#'): continue
				values = line.split()
				if not values: continue
				if values[0] == 'v':
					v = list(map(float, values[1:4]))
					v = [v[0], v[2], v[1]]
					self.vertices.append(v)
				elif values[0] == 'f':
					face = []
					for v in values[1:]:
						w = v.split('/')
						face.append(int(w[0]))
					self.faces.append(face)
	
	def rotate(self, angle):
		for i in range(len(self.vertices)):
			v = self.vertices[i]
			self.vertices[i] = [
				v[0] * math.cos(angle) - v[1] * math.sin(angle),
				v[0] * math.sin(angle) + v[1] * math.cos(angle),
				v[2]
			]

	def BBox(self):
		left = min(self.vertices, key = lambda x: x[0])
		right = max(self.vertices, key = lambda x: x[0])
		front = min(self.vertices, key = lambda x: x[1])
		back = max(self.vertices, key = lambda x: x[1])
		top = min(self.vertices, key = lambda x: x[2])
		bottom = max(self.vertices, key = lambda x: x[2])
		return left[0], right[0], front[1], back[1], top[2], bottom[2]

	def export(self, savePath, filename):
		with open(os.path.join(savePath, filename), 'w') as file:
			file.write('mtllib result.mtl\n')
			file.write('o result\n')
			for v in self.vertices:
				file.write(f'v {v[0]} {v[2]} {v[1]}\n')
			for v in self.texcoords:
				file.write(f'vt {v[0]} {v[1]}\n')
			file.write('s 0\n')
			file.write('usemtl floor\n')
			for i in range(self.triCount):
				# OBJ format uses 1-based indexing
				fs = [str(v + 1) for v in self.faces[i]]
				ts = [str(v + 1) for v in self.texidc[i]]
				file.write(f'f {fs[0]}/{ts[0]} {fs[1]}/{ts[1]} {fs[2]}/{ts[2]}\n')
			file.write('s 1\n')
			file.write('usemtl ceiling\n')
			for i in range(self.triCount, self.triCount * 2):
				# OBJ format uses 1-based indexing
				fs = [str(v + 1) for v in self.faces[i]]
				ts = [str(v + 1) for v in self.texidc[i]]
				file.write(f'f {fs[0]}/{ts[0]} {fs[1]}/{ts[1]} {fs[2]}/{ts[2]}\n')
			file.write('s 2\n')
			file.write('usemtl wall\n')
			for i in range(self.triCount * 2, len(self.faces)):
				# OBJ format uses 1-based indexing
				fs = [str(v + 1) for v in self.faces[i]]
				ts = [str(v + 1) for v in self.texidc[i]]
				file.write(f'f {fs[0]}/{ts[0]} {fs[1]}/{ts[1]} {fs[2]}/{ts[2]}\n')
