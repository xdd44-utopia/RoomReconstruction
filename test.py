import numpy as np
from skimage.feature import local_binary_pattern
from skimage import io, measure, img_as_ubyte
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Load an image
image = img_as_ubyte(io.imread('Test/bottom.png', as_gray=True))

# Compute LBP
radius = 3
n_points = 16 * radius
lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

# Apply thresholding on LBP image for a simple segmentation
# This step might be replaced by clustering in more complex scenarios
thresh = threshold_otsu(lbp_image)
binary_image = lbp_image > thresh

# Label connected components
label_image = measure.label(binary_image)
regions = measure.regionprops(label_image)

# Find the largest region by area
largest_region = max(regions, key=lambda r: r.area)

# Create a mask for the largest region
largest_region_mask = label_image == largest_region.label

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Largest Texture Segment')
plt.imshow(largest_region_mask, cmap='gray')
plt.axis('off')

plt.show()
