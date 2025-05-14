import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d

path = r"C:\Users\USER\Pictures\Screenshots\Screenshot 2025-03-15 161038.png"

# load image 
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# define a kernel (eg sharpening filter)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# apply manual 2D convolution 
filtered = convolve2d(image, kernel, mode = 'same', boundary = 'symm')

# Display result
plt.subplot(1, 2, 1), plt.title("Original"), plt.imshow(image, cmap = 'gray')

plt.subplot(1, 2, 2), plt.title("Filtered (2D convolution)"), plt.imshow(filtered, cmap = 'gray')
plt.show()

