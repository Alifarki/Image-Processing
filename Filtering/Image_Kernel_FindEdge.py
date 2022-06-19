import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("./images/img0.jpg",0)

kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) #پیدا کردن لبه های افقی
kernel2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #پیدا کردن لبه های عمودی
dst = cv2.filter2D(img,-1,kernel)
dst2 = cv2.filter2D(img,-1,kernel2)
plt.subplot(1,3,1),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(dst,cmap='gray'),plt.title('EdgeHorizontal')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(dst2,cmap='gray'),plt.title('EdgeVertical')
plt.xticks([]), plt.yticks([])
plt.show()

