import cv2

import sys

import numpy
from skimage import io, color
from skimage import color as graphcut
import numpy as np
from tkinter import filedialog
from matplotlib import pyplot as plt
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
img = r"D:\Potato_disease\data\Early_blight\0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG"

image = cv2.imread(img)
copy = image.copy()
# Create a mask (of zeros uint8 datatype) that is the same size (width, height) as our original image
mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
x, y , w, h = cv2.selectROI("select the area", image)
start = (x, y)
end = (x + w, y + h)
rect = (x, y , w, h)

cv2.rectangle(copy, start, end, (0,0,255), 3)
imshow("Input Image", copy)
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
image = image * mask2[:,:,np.newaxis]
imshow("Mask", mask * 80)
imshow("Mask2", mask2 * 255)
imshow("Image", image)
cv2.imwrite("1.jpg", image)
rgb = io.imread(r'1.jpg')
img = graphcut.rgb2lab(rgb)
thresholded = np.logical_and(*[img[..., i] > t for i, t in enumerate([40, 0, 0])])
from matplotlib import pyplot as plt
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(rgb);         ax[0].axis('off')
ax[1].imshow(thresholded); ax[1].axis('off')
plt.show()

intersection = numpy.logical_and(img, rgb)
union = numpy.logical_or(img, rgb)
iou_score = numpy.sum(intersection) / numpy.sum(union)
print('IoU is %s' % iou_score)