import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("./images/img4.jpg",0)
n=40
kernel = np.ones((n,n),np.float32)/(n**2) #تعریف کرنل به منظور مات کردن-هرچه بزرگتر باشد کرنل میانگین مات تر
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst,cmap='gray'),plt.title('Average')
plt.xticks([]), plt.yticks([])
plt.show()