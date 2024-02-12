import numpy as np
from PIL import Image

def trim(image):
    image = image.crop(image.getbbox())