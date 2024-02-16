import numpy as np
import math

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
		return v
	return v / norm

def vectorAngleRadians(v1, v2):
	v1 = normalize(v1)
	v2 = normalize(v2)
	return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

def vectorAngleDegrees(v1, v2):
	return vectorAngleRadians * 180 / math.pi


def onSegment(p, q, r):
	return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def orientation(p1, p2, p3):
	ori = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
	if (abs(ori) < 0.0001):
		return 0
	return 1 if ori > 0 else 2

def areLineSegmentIntersect(a1, a2, b1, b2):
	o1 = orientation(a1, a2, b1)
	o2 = orientation(a1, a2, b2)
	o3 = orientation(b1, b2, a1)
	o4 = orientation(b1, b2, a2)
	if (o1 != o2 and o3 != o4):
		return True
	if (o1 == 0 and onSegment(a1, b1, a2)):
		return True
	if (o2 == 0 and onSegment(a1, b2, a2)):
		return True
	if (o3 == 0 and onSegment(b1, a1, b2)):
		return True
	if (o4 == 0 and onSegment(b1, a2, b2)):
		return True
	return False

def isPointInsidePolygon(vertices: np.ndarray, boundary: list, p: list):
		
	upperCount = 0
	lowerCount = 0
	for i in range(len(boundary)):
		a1 = vertices[boundary[i]]
		a2 = vertices[boundary[(i + 1) % len(boundary)]]
		if ((a1[0] < p[0] and a2[0] > p[0]) or (a1[0] > p[0] and a2[0] < p[0])):
			if (a1[1] + (a2[1] - a1[1]) / (a2[0] - a1[0]) * (p[0] - a1[0]) < p[1]):
				upperCount += 1
			if (a1[1] + (a2[1] - a1[1]) / (a2[0] - a1[0]) * (p[0] - a1[0]) > p[1]):
				lowerCount += 1
				
	return upperCount % 2 == 1 and lowerCount % 2 == 1

def isLineSegmentInsidePolygon(vertices: np.ndarray, boundary: list, p1: list, p2: list):

	p1 = np.copy(np.asarray(p1[:]))
	p2 = np.copy(np.asarray(p2[:]))
	d = p2 - p1
	p1 += 0.0075 * d
	p2 -= 0.0075 * d

	if (not isPointInsidePolygon(vertices, boundary, p1) or not isPointInsidePolygon(vertices, boundary, p2)):
		return False

	for i in range(len(boundary)):
		if (areLineSegmentIntersect(vertices[boundary[i]], vertices[boundary[(i + 1) % len(boundary)]], p1, p2)):
			return False

	return True
