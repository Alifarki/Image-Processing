import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("./images/img4.jpg",0)
histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.bar(np.arange(256),histr[:,0],color='b')
plt.show()
equ = cv2.equalizeHist(img)
histr1 = cv2.calcHist([equ],[0],None,[256],[0,256])
plt.bar(np.arange(256),histr1[:,0],color='b')
plt.show()
res = np.hstack((img,equ)) #present 2 image and concat
plt.imshow(res,cmap='gray')
plt.show()