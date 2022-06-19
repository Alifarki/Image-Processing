import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("./images/moon.tif",0)
img=(img-np.min(img)) / (np.max(img)-np.min(img))
kernel =  np.ones((3,3),np.float32) #بدست آوردن همه لبه ها با لاپلاسین
kernel[1,1]=-8
dst = cv2.filter2D(img,-1,kernel)
dst=(dst-np.min(dst)) / (np.max(dst)-np.min(dst))
dst2=img - dst #تقویت کردن و شارپ کردن لبه های تصویر
dst2=(dst2-np.min(dst2)) / (np.max(dst2)-np.min(dst2))
plt.subplot(131),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(dst,cmap='gray'),plt.title('Laplace')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(dst2,cmap='gray'),plt.title('boostingedge')
plt.xticks([]), plt.yticks([])
plt.show()