import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("./images/img0.jpg",0)
hist,bins = np.histogram(img.ravel(),256,[0,256])
plt.bar(np.arange(256),hist,color='g')
plt.show()