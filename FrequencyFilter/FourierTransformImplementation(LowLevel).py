import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('./images/img0.jpg',0)
img=(img-np.min(img))/(np.max(img)-np.min(img))

m = img.shape #اندازه تصویر

ft = np.zeros(img.shape, dtype=np.complex128)  #اندازه F(u,v) که باید اندازه تصویر باشد و حتما نوع داده کامپلکس برای ذخیره سازی مقدار موهومی

for u in range(m[0]):
    for v in range(m[1]):
        for x in range(m[0]):
            for y in range(m[1]): #به ازا همه uوv ها و همه پیکسل های تصویر
                ft[u,v] = ft[u,v] + img[x,y] * np.exp(-2*np.pi*1j * (x*u/m[0] + y*v/m[1]))
    print(u)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(ft, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
