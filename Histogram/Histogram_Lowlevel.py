import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("./images/img0.jpg",0)
histr=np.zeros(256)
for i in range(256):
    idx=np.sum(img==i)
    histr[i]=idx
plt.bar(np.arange(256),histr,color='g')
plt.show()