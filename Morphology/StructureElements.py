import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread('./images/j.png',0)
# plt.imshow(img)
# plt.show()

se=cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4)) #plus shape
print(se)

se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)) #circle shape
print(se)

se=cv2.getStructuringElement(cv2.MORPH_RECT,(9,20)) #rectangle shape
print(se)