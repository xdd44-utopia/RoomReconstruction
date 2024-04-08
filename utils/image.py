import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def trim(image):
    image = image.crop(image.getbbox())

def getAverage(image):
    image = np.array(image)[:, :, :-1]
    return image.mean(axis=0).mean(axis=0)

def getDominant(image, get_palette = False):
    image = np.array(image)[:, :, :-1]
    pixels = np.float32(image.reshape(-1, 3))

    n_colors = 16
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    palette = np.uint8(palette)
    for i in range(len(counts)):
        if ((palette[i] == np.array([0, 0, 0], dtype = np.uint8)).all()):
            palette = np.delete(palette, i, 0)
            counts = np.delete(counts, i)
            break
    indices = np.argsort(counts)[::-1]
    freqs = np.hstack(counts[indices]/float(counts.sum()))

    # Weighted average of top three dominant colors
    dominant = palette[indices[0]] * freqs[0] / (freqs[0] + freqs[1] + freqs[2]) + palette[indices[1]] * freqs[1] / (freqs[0] + freqs[1] + freqs[2]) + palette[indices[2]] * freqs[2] / (freqs[0] + freqs[1] + freqs[2])
    if (get_palette):
        return dominant, palette
    else:
        return dominant

def plotAverageDominant(image):
    image = np.array(image)[:, :, :-1]
    pixels = np.float32(image.reshape(-1, 3))

    average = image.mean(axis=0).mean(axis=0)

    n_colors = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    palette = np.uint8(palette)
    for i in range(len(counts)):
        if ((palette[i] == np.array([0, 0, 0], dtype = np.uint8)).all()):
            palette = np.delete(palette, i, 0)
            counts = np.delete(counts, i)
            break
    indices = np.argsort(counts)[::-1]
    freqs = np.hstack(counts[indices]/float(counts.sum()))
    avg_patch = np.ones(shape=(500, 500, 3), dtype=np.uint8)*np.uint8(average)

    indices = np.argsort(counts)[::-1]   
    freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
    rows = np.int_(500*freqs)

    dom_patch = np.zeros(shape=(500, 500, 3), dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
    ax0.imshow(avg_patch)
    ax0.set_title('Average color')
    ax0.axis('off')
    ax1.imshow(dom_patch)
    ax1.set_title('Dominant colors')
    ax1.axis('off')
    plt.show()

# with Image.open("../Test/bottom.png") as im:
#     plotAverageDominant(im)