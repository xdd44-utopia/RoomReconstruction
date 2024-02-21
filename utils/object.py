import pygame
from OpenGL.GL import *
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import numpy as np

from utils.fit_plane import extract_planes
from utils.mesh import extractConvexBoundary

def MTL(filename):
	contents = {}
	mtl = None
	for line in open(filename, "r"):
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
			surf = pygame.image.load(mtl['map_Kd'])
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
	def __init__(self, filename, swapyz=False):
		"""Loads a Wavefront OBJ file. """
		self.vertices = []
		self.normals = []
		self.texcoords = []
		self.faces = []

		material = None
		print("Loading Geometry...")
		for line in tqdm(open(filename, "r")):
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
				self.mtl = MTL(values[1])
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
			self.vertices[i] = v[0] - (bbox[1] - bbox[0]) / 2, v[1] - (bbox[3] - bbox[2]) / 2, v[2] - bbox[4]
	
	def orient(self):
		vertices = [list(v) for v in self.vertices]
		sample_points = np.random.choice(len(vertices), int(len(vertices) / 10)).tolist()
		plane_eqs, plane_points, remaining_points = extract_planes(np.array(vertices)[sample_points])

		norm = plane_eqs[0]
		i = 0
		while (norm[2] > 0.01 and i < len(plane_eqs)):
			norm = plane_eqs[i]
			i += 1
		theta = math.atan(norm[1] / norm[0])

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
	def __init__(self, filename = "", vertices = [], faces = []):
		"""Loads a Wavefront OBJ file. """
		self.vertices = vertices
		self.faces = faces
	
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

	def export(self, filename):
		with open(filename, 'w') as file:
			for v in self.vertices:
				file.write(f'v {v[0]} {v[1]} {v[2]}\n')
			for face in self.faces:
				# OBJ format uses 1-based indexing
				face_indices = [str(v) for v in face]
				file.write(f'f {" ".join(face_indices)}\n')
