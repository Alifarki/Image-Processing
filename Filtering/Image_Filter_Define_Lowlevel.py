import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./images/img0.jpg', 0)
img = (img - np.min(img)) / (np.max(img) - np.min(img))

# kernel = np.ones((3,3), np.float32)
# kernel[1,1] = -8
# dst = cv2.filter2D(img,-1,kernel)
# dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst))
# dst = img - dst
# dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst))
# plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst, cmap='gray'),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

n = 11
kernel = np.ones((n,n), np.float32) / n**2
s = np.int((n - 1) / 2)
kernel = np.fliplr(np.flipud(kernel))
mn = img.shape
img = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, 0)
img_filter = np.zeros(img.shape)
for i in range(s, mn[0]-s):
    for j in range(s, mn[1]-s):
        img_filter[i,j] = np.sum(img[i-s:i+s+1,j-s:j+s+1] * kernel)

img_filter = img_filter[s:mn[0]-s, s:mn[1]-s]
plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_filter, cmap='gray'),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()