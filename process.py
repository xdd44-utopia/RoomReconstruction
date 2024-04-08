import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import sys
import pygame
from pygame.locals import *
from pygame.constants import *

import numpy as np
from skimage import io
from PIL import Image

from OpenGL.GL import *
from OpenGL.GLU import *

from utils.object import OBJ, NakedOBJ
from utils.mesh import triangulizePolygon, delaunayTriangulizePolygon, extractConvexBoundary, extrudeBoundary
from utils.plot import *
from utils.image import getAverage, getDominant, plotAverageDominant

def boundingOrtho(display, bbox, view):
	glLoadIdentity()

	ml, mr, mt, mb = None, None, None, None
	if (view == "top"):
		ml, mr, mt, mb, _, _ = bbox
	elif (view == "bottom"):
		ml, mr, mt, mb, _, _ = bbox
		mt, mb = -mb, -mt
	elif (view == "front"):
		ml, mr, _, _, mt, mb, = bbox
		ml, mr = -mr, -ml
	elif (view == "back"):
		ml, mr, _, _, mt, mb, = bbox
	elif (view == "left"):
		_, _, ml, mr, mt, mb, = bbox
		ml, mr = -mr, -ml
	elif (view == "right"):
		_, _, ml, mr, mt, mb, = bbox

	if ((mr - ml) / (mb - mt) < display[0] / display[1]):
		w = display[0] / display[1] * (mb - mt)
		padding = (w - (mr - ml)) / 2
		# glOrtho(ml - padding, mr + padding, mt, mb, 0.1, 50)
		glOrtho(ml - padding, mr + padding, mt, mb, 0.1, 1000)
	else:
		h = display[1] / display[0] * (mr - ml)
		padding = (h - (mb - mt)) / 2
		glOrtho(ml, mr, mt - padding, mb + padding, 0.1, 1000)
	
	if (view == "top"):
		glTranslatef(0, 0, -500)
	elif (view == "bottom"):
		glTranslatef(0, 0, -500)
		glRotatef(180, 1, 0, 0)
	elif (view == "front"):
		glRotatef(90, 1, 0, 0)
		glRotatef(180, 0, 1, 0)
		glTranslatef(0, -500, 0)
	elif (view == "back"):
		glRotatef(-90, 1, 0, 0)
		glTranslatef(0, 500, 0)
	elif (view == "left"):
		glTranslatef(0, 0, -500)
		glRotate(-90, 1, 0, 0)
		glRotate(90, 0, 0, 1)
	elif (view == "right"):
		glTranslatef(0, 0, -500)
		glRotate(-90, 1, 0, 0)
		glRotate(-90, 0, 0, 1)

def captureTexture(display, model, view):
	pixels = (GLubyte * (4 * display[0] * display[1]))(0)
	glReadPixels(0, 0, display[0], display[1], GL_RGBA, GL_UNSIGNED_BYTE , pixels)
	image = Image.frombytes(mode="RGBA", size=(display[0], display[1]), data = pixels)
	image = image.transpose(Image.FLIP_TOP_BOTTOM)
	image = image.crop(image.getbbox())
	# Ensure it's still 720 height
	width = int((float(image.size[0]) * 720.0 / float(image.size[1])))
	image = image.resize((width, 720))
	image.save(f"Test/{view}.png")
	return image

def exportTexture(viewCaptures):
	ceiling = np.ones(shape=(128, 128, 3), dtype=np.uint8) * np.uint8(getDominant(viewCaptures[0]))
	io.imsave("./Result/ceiling.jpg", ceiling)
	floor = np.ones(shape=(128, 128, 3), dtype=np.uint8) * np.uint8(getAverage(viewCaptures[1]))
	io.imsave("./Result/floor.jpg", floor)
	wallImages = viewCaptures[2:]
	totalWidth = sum(img.width for img in wallImages)
	wallOrigin = Image.new('RGBA', (totalWidth, 720))
	offset = 0
	for img in wallImages:
		wallOrigin.paste(img, (offset, 0))
		offset += img.width
	wallOrigin.show()
	wall = np.ones(shape=(128, 128, 3), dtype=np.uint8) * np.uint8(getDominant(wallOrigin))
	io.imsave("./Result/wall.jpg", wall)


def main():

	np.random.seed(0)

	pygame.init()

	display = (1280, 720)
	pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

	model = OBJ("Scan", "test1.obj", swapyz=True)
	BBox = model.BBox()

	print("Extract boundary...")
	vertices, boundary = extractConvexBoundary(
		np.asarray(model.vertices),
		sampleRate = 0.1,
		distLimit = 20
	)

	print("Construct bottom faces...")
	triangles = delaunayTriangulizePolygon(vertices, boundary)
	triangleCount = len(triangles)
	
	print("Extrude room...")
	vertices, triangles, texcoords, texidc = extrudeBoundary(vertices, boundary, triangles, model.height())
	result = NakedOBJ(vertices = vertices, faces = triangles, texcoords = texcoords, texidc = texidc, bottomTriangleCount = triangleCount)
	result.export("Result", "result.obj")

	#gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

	glEnable(GL_DEPTH_TEST)
	#glEnable(GL_CULL_FACE)
	glCullFace(GL_FRONT)

	views = ["top", "bottom", "front", "back", "left", "right"]
	viewCaptures = []

	while True:
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_KP_ENTER:
					pass
				if event.key == pygame.K_a:
					glTranslatef(0.2, 0, 0)
				if event.key == pygame.K_d:
					glTranslatef(-0.2, 0, 0)
				if event.key == pygame.K_w:
					glTranslatef(0, 0, -0.2)
				if event.key == pygame.K_s:
					glTranslatef(0, 0, 0.2)
				if event.key == pygame.K_UP:
					glRotate(30, 1, 0, 0)
				if event.key == pygame.K_DOWN:
					glRotate(-30, 1, 0, 0)
				if event.key == pygame.K_LEFT:
					glRotate(30, 0, 0, 1)
				if event.key == pygame.K_RIGHT:
					glRotate(-30, 0, 0, 1)

			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		
		glPushMatrix()
		view = None
		if (len(views) > 0):
			view = views[0]
			boundingOrtho(display, BBox, view)
		glCallList(model.gl_list)
		glPopMatrix()

		if (len(views) > 0):
			viewCaptures.append(captureTexture(display, model, view))
			print(view)
			views.pop(0)
		else:
			exportTexture(viewCaptures)
			pygame.display.quit()
			pygame.quit()
			sys.exit()

		pygame.display.flip()
		pygame.time.wait(10)


main()