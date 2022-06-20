import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread('./images/j.png',0)
plt.imshow(img)
plt.show()

se=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
erosion=cv2.morphologyEx(img,cv2.MORPH_ERODE,se)
plt.imshow(erosion)
plt.show()

beta=img-erosion
plt.imshow(beta)
plt.show()