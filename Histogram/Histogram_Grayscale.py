import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./images/img0.jpg",0)
histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.bar(np.arange(256),histr[:,0],color='b')
plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('home.jpg',0)
# plt.hist(img.ravel(),256,[0,256]);
# plt.show()