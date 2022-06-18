import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("./images/img4.jpg",0)
MN=img.size
histr=cv2.calcHist([img],[0],None,[256],[0,256])
pr=histr[:,0]/MN
s=255*np.cumsum(pr)
img_equalize=np.zeros(img.shape)
for i in range(255):
    img_equalize[img==i] = np.round(s[i])
plt.figure(2)
plt.imshow(img_equalize,cmap='gray')
plt.figure(3)
plt.plot(s)
plt.show()
