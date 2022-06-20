import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread('./images/j.png',0)
plt.imshow(img)
plt.show()

n,label=cv2.connectedComponents(img) #تعداد جزیره های ما را نمایش می دهد که سه جزیره را نمایش داده است


#label[label==2]=0 #ابجکت شماره 2 را صفر میکند
# thresh=200
# for i in range(n):
#     if i > 0:
#         c=np.sum(label==i)
#         if c<thresh:  #آبجکت های با کمتر از 200 پیکسل حذف میشوند و با پس زمینه کسان میشوند
#             label[label==i]=0
#
# plt.imshow(label)
# plt.show()

thresh=200
for i in range(n):
    if i > 0:
        x,y,w,h=cv2.boundingRect(np.uint8(label==i)) #بدست آوردن نقطه شروع ابجکت و ارتفاع و پهنا
        aspect_ratio=float(w)/h
        if aspect_ratio > 1.1 or aspect_ratio < 0.9: #نشان دهنده دایره
            label[label==i]=0

plt.imshow(label)
plt.show()


