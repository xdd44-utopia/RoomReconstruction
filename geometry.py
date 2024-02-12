import pygame
from OpenGL.GL import *
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import numpy as np

from config import distLimit
from utils.fit_plane import extract_planes

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
			i += 1
			norm = plane_eqs[i]
		theta = math.atan(norm[1] / norm[0])

		print("Rotating...")
		self.rotate(-theta)
		print("Centering...")
		self.center()

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

class V:
	def __init__(self, v):
		self.x = v[0]
		self.y = v[1]

	def magnitude(self):
		return math.sqrt(self.x ** 2 + self.y ** 2)
	
	def distance(self, other):
		v = V((other.x - self.x, other.y - self.y))
		return v.magnitude()

	def toString(self):
		return f"({self.x}, {self.y})"

def angleLineAxis(v1, v2):
	if (v2.x - v1.x == 0):
		return math.pi / 2 if v2.y > v1 else - math.pi / 2
	else:
		k = (v2.y - v1.y) / (v2.x - v1.x)
		angle = math.atan(k)
		if (angle > 0 and v1.y > v2.y):
			angle -= math.pi
		if (angle < 0 and v1.y < v2.y):
			angle += math.pi
		return angle

def angleTwoVectors(s1, t1, s2, t2):
	v1 = V((t1.x - s1.x, t1.y - s1.y))
	v2 = V((t2.x - s2.x, t2.y - s2.y))
	if (v1.magnitude() == 0 or v2.magnitude() == 0):
		return 0
	angle = math.acos((v1.x * v2.x + v1.y * v2.y) / (v1.magnitude() * v2.magnitude()))
	if (v1.x * v2.y - v1.y * v2.x < 0):
		angle = - angle
	return angle

def extractBoundary(model):
	
	vertices = [V(v) for v in model.vertices]

	vertices.sort(key = lambda v: v.x)

	vertices.sort(key = lambda v: v.x)
	result = [vertices[0]]
	vertices.pop(0)
	vertices.sort(key = lambda v: angleLineAxis(result[0], v))
	result.append(vertices[0])
	cur = vertices.pop(0)
	vertices.append(result[0])

	# for ite in tqdm(range(len(vertices))):
	while (cur != result[0]):
		vertices.sort(key = lambda v: angleTwoVectors(result[-2], result[-1], result[-1], v))
		for i in range(len(vertices)):
			if (result[-1].distance(vertices[i]) < distLimit and angleTwoVectors(result[-2], result[-1], result[-1], vertices[i]) > - math.pi / 2):
				cur = vertices.pop(i)
				result.append(cur)
				break
		if (cur == result[0]):
			break

	return result

if __name__ == "__main__":

	model = NakedOBJ("testOrient.obj")
	
	sample_points = np.random.choice(len(model.vertices), int(len(model.vertices) / 500)).tolist()

	x = np.array([v[0] for v in np.array(model.vertices)[sample_points]])
	y = np.array([v[1] for v in np.array(model.vertices)[sample_points]])
	plt.figure(figsize=(10,10))
	plt.scatter(x, y, marker='o')

	sample_points = np.random.choice(len(model.vertices), int(len(model.vertices) / 10)).tolist()
	plane_eqs, plane_points, remaining_points = extract_planes(np.array(model.vertices)[sample_points])
	print("Plane extracted")

	norm = plane_eqs[0]
	i = 0
	while (norm[2] > 0.01 and i < len(plane_eqs)):
		i += 1
		norm = plane_eqs[i]
	theta = math.atan(norm[1] / norm[0])
	model.rotate(-theta)
	print("Model rotated")

	sample_points = np.random.choice(len(model.vertices), int(len(model.vertices) / 500)).tolist()

	x = np.array([v[0] for v in np.array(model.vertices)[sample_points]])
	y = np.array([v[1] for v in np.array(model.vertices)[sample_points]])
	plt.scatter(x, y, marker='^')

	plt.show()

	quit()

	boundary = extractBoundary(model)

	resultOBJ = NakedOBJ()
	resultFaces = []
	for i in range(1, len(boundary) - 1):
		resultFaces.append([len(boundary) - 1, i - 1, i])
	
	resultOBJ.vertices = [[v.x, v.y, 0] for v in boundary]
	resultOBJ.faces = resultFaces
	resultOBJ.export("testResult.obj")

	x = np.array([v.x for v in boundary])
	y = np.array([v.y for v in boundary])
	plt.figure(figsize=(10,10))
	plt.scatter(x, y)
	plt.savefig('100.jpg')
		
	print(len(boundary))