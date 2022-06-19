import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./images/img4.jpg",0)

blur = cv2.bilateralFilter(img,9,75,75) #بسیار خوب برای حذف نویز ها

plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur,cmap='gray'),plt.title('BlurBilateral')
plt.xticks([]), plt.yticks([])
plt.show()