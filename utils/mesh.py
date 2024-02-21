import math
import numpy as np
import matplotlib.pyplot as plt
from utils.vector import normalize, vectorAngleRadians, vectorAngleDegrees, isLineSegmentInsidePolygon, angleLineAxis, angleTwoVectors
from utils.plot import *

def clockwiseBoundary(vertices: np.ndarray, boundary: list, localNormal: list):

		angleAcc = 0
		for i in range(len(boundary)):
			va = normalize(vertices[boundary[i]] - vertices[boundary[(i + len(boundary) - 1) % len(boundary)]])
			vb = normalize(vertices[boundary[(i + 1) % len(boundary)]] - vertices[boundary[i]])
			isConvex = vectorAngleRadians(normalize(np.cross(va, vb)), localNormal) < math.pi / 4
			angleAcc += (math.pi - vectorAngleRadians(va, vb)) * (1 if isConvex else -1)
		
		if (angleAcc < 0):
			newBoundary = []
			for i in range(len(boundary) - 1, -1, -1):
				newBoundary.append(boundary[i])
			return newBoundary
		else:
			return boundary

def simplifyBoundary(vertices: np.ndarray, orignalBoundary: list):
	boundary = orignalBoundary[:]
	while (True):
		prevCount = len(boundary)
		for i in range(len(boundary)):
			prev = (i + len(boundary) - 1) % len(boundary)
			next = (i + 1) % len(boundary)
			if (np.linalg.norm(np.cross(vertices[boundary[next]] - vertices[boundary[i]], vertices[boundary[i]] - vertices[boundary[prev]])) < 0.001):
				del boundary[i]
				break
		if (len(boundary) == prevCount):
			break
	return boundary

def splitMonotonePolygon(vertices: np.ndarray, boundary: list):

		vertexPointers = []
		for i in range(len(boundary)):
			vertexPointers.append(i)
			
		vertexPointers.sort(key = lambda x: vertices[boundary[x]][0])

		edges = []
		for i in range(len(vertexPointers)):
			cur = vertexPointers[i]
			pre = (cur + len(boundary) - 1) % len(boundary)
			nex = (cur + 1) % len(boundary)

			if (normalize(vertices[boundary[pre]] - vertices[boundary[cur]])[1] > normalize(vertices[boundary[nex]] - vertices[boundary[cur]])[1]):
				pre, nex = nex, pre

			if (vertices[boundary[pre]][0] > vertices[boundary[cur]][0] and vertices[boundary[nex]][0] > vertices[boundary[cur]][0]):
				for j in range(i - 1, -1, -1):
					if (isLineSegmentInsidePolygon(vertices, boundary, vertices[boundary[vertexPointers[i]]], vertices[boundary[vertexPointers[j]]])):
						edges.append(min(boundary[vertexPointers[i]], boundary[vertexPointers[j]]))
						edges.append(max(boundary[vertexPointers[i]], boundary[vertexPointers[j]]))
						break
						
			if (vertices[boundary[pre]][0] <= vertices[boundary[cur]][0] and vertices[boundary[nex]][0] <= vertices[boundary[cur]][0]):
				for j in range(i + 1, len(vertexPointers)):
					if (isLineSegmentInsidePolygon(vertices, boundary, vertices[boundary[vertexPointers[i]]], vertices[boundary[vertexPointers[j]]])):
						edges.append(min(boundary[vertexPointers[i]], boundary[vertexPointers[j]]))
						edges.append(max(boundary[vertexPointers[i]], boundary[vertexPointers[j]]))
						break
						
		return splitBoundaryByEdges(vertices, boundary, edges, [0, 0, 1])

def splitBoundaryByEdges(vertices: np.ndarray, boundary: list, edges: list, localNormal: list):
# this only works for clockwised boundary
# edges[i * 2] and edges[i * 2 + 1] forms an edge to split

		boundary = clockwiseBoundary(vertices, boundary, localNormal)
		if (len(edges) == 0):
			return [boundary]

		adjList = [[] for i in range(len(vertices))]
		# each connection consists of [edge, visitCount, clockwise]

		for e in range(len(boundary)):
			u = boundary[e]
			v = boundary[(e + 1) % len(boundary)]
			adjList[u].append([v, 1, 1])
			adjList[v].append([u, 1, -1])
		
		for i in range(len(edges) // 2):
			adjList[edges[i * 2]].append([edges[i * 2 + 1], 2, 0])
			adjList[edges[i * 2 + 1]].append([edges[i * 2], 2, 0])

		newBoundaries = []
		while (True):
			cur = [-1, -1] #[u, v]
			for i in range(len(vertices)):
				for j in range(len(adjList[i])):
					if (adjList[i][j][2] == 1 and len(adjList[i]) == 2):
						cur = [i, adjList[i][j][0]]
					elif (adjList[i][j][2] == -1 and len(adjList[adjList[i][j][0]]) == 2):
						cur = [adjList[i][j][0], i]
					if (cur[0] != -1):
						break
				if (cur[0] != -1):
					break
			if (cur[0] == -1):
				break

			newBoundary = []
			start = cur[0]
			newBoundary.append(cur[0])
			while (True):
				i = 0
				while i < len(adjList[cur[0]]):
					if (adjList[cur[0]][i][0] == cur[1]):
						adjList[cur[0]][i] = (adjList[cur[0]][i][0], adjList[cur[0]][i][1] - 1, adjList[cur[0]][i][2])
						if (adjList[cur[0]][i][1] == 0):
							del adjList[cur[0]][i]
					i += 1
				
				i = 0
				while i < len(adjList[cur[1]]):
					if (adjList[cur[1]][i][0] == cur[0]):
						adjList[cur[1]][i] = (adjList[cur[1]][i][0], adjList[cur[1]][i][1] - 1, adjList[cur[1]][i][2])
						if (adjList[cur[1]][i][1] == 0):
							del adjList[cur[1]][i]
					i += 1
							
				if (cur[1] == start):
					break
				else:
					newBoundary.append(cur[1])

				maxEdge = [-2147483647, -1] #[max angle, index]
				v1 = vertices[cur[1]] - vertices[cur[0]]
				for i in range(len(adjList[cur[1]])):
					nex = adjList[cur[1]][i][0]
					if (nex != cur[0]):
						v2 = vertices[nex] - vertices[cur[1]]
						angle = vectorAngleRadians(v1, v2) * (1 if vectorAngleRadians(normalize(np.cross(v1, v2)), localNormal) < math.pi / 4 else -1)
						if (angle > maxEdge[0]):
							maxEdge = [angle, nex]
				cur[0] = cur[1]
				cur[1] = maxEdge[1]
				
			newBoundaries.append(newBoundary)

		return newBoundaries

def triangulizeMonotonePolygon(vertices: np.ndarray, boundary: list):
# returns a list of triangle indices form a triangle

		if (len(boundary) == 3):
			return boundary

		#Algorithm:
		#Sort vertices by x
		#From left to right, connect each vertex with all vertices at the left side of it, and trim the formed triangles off the polygon

		#Use prev to keep track of its left adjacent vertex
		#Note that the first vertex has both adjacent vertices on its right, and the last vertex has both on its left

		#############
		#Translated from C#, 0 = index, 1 = prev, 2 = next, 3 = removed
		vertexPointers = []
		removed = []
		for i in range(len(boundary)):

			pre = boundary[(i + len(boundary) - 1) % len(boundary)] if vertices[boundary[(i + len(boundary) - 1) % len(boundary)]][0] < vertices[boundary[(i + 1) % len(boundary)]][0] else boundary[(i + 1) % len(boundary)]

			nex = boundary[(i + len(boundary) - 1) % len(boundary)] if vertices[boundary[(i + len(boundary) - 1) % len(boundary)]][0] > vertices[boundary[(i + 1) % len(boundary)]][0] else boundary[(i + 1) % len(boundary)]
			
			vertexPointers.append({'index': boundary[i], 'prev': pre, 'next': nex, 'removed': False})

		vertexPointers.sort(key = lambda x: vertices[x['index']][0])
		
		triangles = []

		#A polygon with n vertices will be splitted n - 3 times
		for s in range(len(boundary) - 3):
			count = 0
			leftMost = [0, 0, 0]
			for i in range(len(vertexPointers)):
				if (not vertexPointers[i]['removed']):
					leftMost[count] = i
					count += 1
				if (count == 3):
					break

			found = False
			for i in range(leftMost[2], len(vertexPointers)):
				if (not vertexPointers[i]['removed']):
					pre = next((t for t in range(len(vertexPointers)) if vertexPointers[t]['index'] == vertexPointers[i]['prev']))
					prepre = leftMost[1] if pre == leftMost[0] else next((t for t in range(len(vertexPointers)) if vertexPointers[t]['index'] == vertexPointers[pre]['prev']))
					if (isLineSegmentInsidePolygon(vertices, boundary, vertices[vertexPointers[i]['index']], vertices[vertexPointers[prepre]['index']])):
						triangles.append([
							vertexPointers[pre]['index'],
							vertexPointers[prepre]['index'],
							vertexPointers[i]['index']
						])
						vertexPointers[pre] = {'index': -1, 'prev': -1, 'next': -1, 'removed': True}
						vertexPointers[i] = {
							'index': vertexPointers[i]['index'],
							'prev': vertexPointers[prepre]['index'],
							'next': vertexPointers[i]['next'],
							'removed': False
						}
						found = True
						break
					
					if (i == len(vertexPointers) - 1):
						pre = next((t for t in range(len(vertexPointers)) if vertexPointers[t]['index'] == vertexPointers[i]['next']))
						prepre = leftMost[1] if pre == leftMost[0] else next((t for t in range(len(vertexPointers)) if vertexPointers[t]['index'] == vertexPointers[pre]['prev']))
						if (isLineSegmentInsidePolygon(vertices, boundary, vertices[vertexPointers[i]['index']], vertices[vertexPointers[prepre]['index']])):
							triangles.append([
								vertexPointers[pre]['index'],
								vertexPointers[prepre]['index'],
								vertexPointers[i]['index']
							])
							vertexPointers[pre] = {'index': -1, 'prev': -1, 'next': -1, 'removed': True}
							vertexPointers[i] = {
								'index': vertexPointers[i]['index'],
								'prev': vertexPointers[i]['prev'],
								'next': vertexPointers[prepre]['index'],
								'removed': False
							}
							found = True
							break
							
			if (not found):
				print("Failed on ", leftMost, len(vertexPointers))
				
		#The remaining vertices form the last triangle
		remainingVertices = []
		for i in range(len(vertexPointers)):
			if (not vertexPointers[i]['removed']):
				remainingVertices.append(vertexPointers[i]['index'])
		triangles.append(remainingVertices)

		return triangles

def triangulizePolygon(vertices: np.ndarray, boundary: list):
	monotonePolygons = splitMonotonePolygon(vertices, boundary)
	triangles = []
	for polygon in monotonePolygons:
		triangles.extend(triangulizeMonotonePolygon(vertices, polygon))
	return triangles



def extractConvexBoundary(vertices: np.ndarray, sampleRate = 0.1, distLimit = 100):

	samples = np.random.choice(vertices.shape[0], int(vertices.shape[0] * sampleRate), replace = False)
	verticesSampled = np.asarray([[v[0], v[1], 0] for v in vertices[samples]])

	vertexPointers = []
	for i in range(len(verticesSampled)):
		vertexPointers.append(i)
		
	vertexPointers.sort(key = lambda v: verticesSampled[v][0])
	result = [vertexPointers[0]]
	pre = vertexPointers.pop(0)

	vertexPointers.sort(key = lambda v: angleLineAxis(verticesSampled[pre], verticesSampled[v]))
	result.append(vertexPointers[0])
	cur = vertexPointers.pop(0)
	vertexPointers.append(result[0])

	ite = 0
	found = False

	while (cur != result[0]):
		found = False
		vertexPointers.sort(key = lambda v: angleTwoVectors(verticesSampled[cur] - verticesSampled[pre], verticesSampled[v] - verticesSampled[cur]))
		
		for i in range(len(vertexPointers)):
			dist = np.linalg.norm(verticesSampled[cur] - verticesSampled[vertexPointers[i]])
			angle = angleTwoVectors(verticesSampled[cur] - verticesSampled[pre], verticesSampled[vertexPointers[i]] - verticesSampled[cur])
			if (dist < distLimit and angle > - math.pi / 2):
				pre = cur
				cur = vertexPointers.pop(i)
				result.append(cur)
				found = True
				break
				
		ite += 1
		if (ite > 1000 or not found):
			break

	del result[0]
	return verticesSampled, result