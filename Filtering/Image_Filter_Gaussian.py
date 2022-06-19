import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./images/img4.jpg",0)

blur = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur,cmap='gray'),plt.title('Blur')
plt.xticks([]), plt.yticks([])
plt.show()