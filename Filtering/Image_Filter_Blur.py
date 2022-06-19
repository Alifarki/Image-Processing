import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./images/img4.jpg",0)

blur = cv2.blur(img,(30,30)) #شبیه یک کرنل عمل میکند که هر چه بزرگتر بشه کرنل مات تر و smoothتر میکند تصویر را

plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur,cmap='gray'),plt.title('Blur')
plt.xticks([]), plt.yticks([])
plt.show()