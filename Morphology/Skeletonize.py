import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


img=cv2.imread('./images/2.png',0)
img[img == 255] = 1 #در تصویر سیاه و سفید اونایی که برابر 255 هستند یک میکند
plt.imshow(img)
plt.show()
skeleton = skeletonize(img)
plt.imshow(skeleton)
plt.show()
