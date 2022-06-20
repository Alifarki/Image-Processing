import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./images/Fig1027(a)(van_original).tif',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #یک آستانه گذاری معمولی با مقدار 127 انتخاب شده است.

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,11,2) #cv2,thresh_binaryنوع آستانه گیری
#توی Adaptive بدون اینکه خود ما پیدا کنیم استانه رو پیدا میکند براساس میانگین گیری
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


#آستانه گذاری با otsu هم امکان پذیر است