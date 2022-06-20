import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./images/Fig1004(b)(turbine_blade_black_dot).tif', 0)
img = (img - np.min(img)) / (np.max(img) - np.min(img))

w=np.array([[1, 1, 1],
              [1,-8, 1],
              [1, 1, 1]],np.float32) #فیلتر میانگین

dst = cv2.filter2D(img, -1, w) #اعمال فیلتر میانگین
t = np.max(dst) #threshold #بیشترین مقدار بعد اعمال فیلتر میانگین
dst = dst >= t #مقدار dst هایی که از آستانه بیشتر هستند را نمایش بده
plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst, cmap='gray'),plt.title('Averaging')
plt.show()