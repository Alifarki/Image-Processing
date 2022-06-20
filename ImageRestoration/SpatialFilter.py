#فیلترهای مکانی

import numpy as np
import matplotlib.pyplot as plt
import cv2



img = cv2.imread('./images/img2.jpg', 0)
img = (img - np.min(img)) / (np.max(img) - np.min(img))
rows, cols = img.shape

plt.figure(0)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
plt.close(0)



ksize = 3 #سایزفیلتر
padsize = np.int_((ksize - 1)/2) #اندازه padding
img_pad = cv2.copyMakeBorder(img, padsize, padsize, padsize, padsize, cv2.BORDER_DEFAULT) #top,bottom,left,right and bordertype
img_geometric = np.zeros_like(img) #میانگین هندسی یا geometric
for r in range(rows):
    for c in range(cols): #برای انجام conv
        img_geometric[r,c] = np.prod(img[r:r+ksize,c:c+ksize]) ** (1/(ksize**2)) #اسکن کردن تصویر و فیلتر
        #درایه های موجود در ماتریس در هم ضرب شوندprod
        #میانگین هندسی موجود در اسلایدها
img_geometric = (img_geometric - np.min(img_geometric)) / (np.max(img_geometric) - np.min(img_geometric))

plt.figure(1)
plt.imshow(img_geometric, cmap='gray')
plt.axis('off')
plt.show()
plt.close(0)