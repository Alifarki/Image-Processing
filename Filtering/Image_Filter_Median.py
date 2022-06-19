import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./images/img4.jpg",0)

median = cv2.medianBlur(img,5)

plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median,cmap='gray'),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()


median = cv2.medianBlur(img,5)
