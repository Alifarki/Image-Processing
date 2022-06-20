import cv2
import numpy as np
from matplotlib import pyplot as plt

img=np.zeros((256,256),np.uint8)
img[100:200,100:200]=255
plt.imshow(img)
plt.show()

sobelx8u = cv2.Sobel(img,cv2.CV_8U,1 ,0,ksize=5) #سوبل خودش یک اپراتور گرادیان افقی و عمودی که مقادیر 1 و 0 یعنی در جهت x,y فعال باشد یا خیر
#نوع دیتاتایپ از نوع CV_8U که همون uint8 ---خروجی ماست
#اندازه کرنل سایزکه 5*5 است
#بخاطر مقادیر ستونی فیلتر سوبل و uint8 بودن ضرب ها فقط لبه اولی ستونی را بدست می آورد.پس از flot استفاده میکنیم
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f) #مثبت کردن مقادیر منفی که همون لبه سمت راست منفی بود و سیاه شد
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()



#canny edge detection
img = cv2.imread('./images/Fig1027(a)(van_original).tif',0)
img = cv2.resize(img, (100,100))
edges = cv2.Canny(img, 150, 220)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

gx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
gy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
mag = np.sqrt(gx**2+gy**2) #magnitude یا همون M

plt.imshow(img,cmap = 'gray')
plt.quiver(range(img.shape[1]), range(img.shape[0]), gx, gy)
plt.show()