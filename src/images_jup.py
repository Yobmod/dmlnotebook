# ***************Method 1: Using PILLOW (PIL)****************

# importing PIL Module
import cv2 as cv
import imageio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


img: Image = Image.open('./earth-globe.jpg')
img.show()


# #***************Method 2: Using matplotlib****************


img: np.ndarray = mpimg.imread('./earth-globe.jpg')
plt.imshow(img)
plt.show()

# ***************Method 3: Using imageio and matplotlib****************

img: np.ndarray = imageio.imread('./earth-globe.jpg')
plt.imshow(img)
plt.show()

# ***************Method 4: Using OpenCV-Python****************

img: np.ndarray = cv.imread('./earth-globe.jpg', -1)

# to display image in pop-up until you press any key
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# ***************Method 5: Using OpenCV-matpotlib****************


img = cv.imread('./earth-globe.jpg', 1)

# convert BRG to RGB (opencv -> matplotlib)
RGBimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(RGBimg)
plt.show()

