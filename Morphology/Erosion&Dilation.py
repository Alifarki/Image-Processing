import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('./images/j.png',0)
plt.imshow(img)
plt.show()
se=np.ones((5,5),np.uint8)#مربع تمام یک که همان SE
# se=np.ones((5,1),np.uint8)#مربع تمام یک که همان SE-ستونی و اشکال دیگر
# se[4,0]=0
erosion=cv2.erode(img,se)
plt.imshow(erosion)
plt.show()

dilation=cv2.dilate(img,se)
plt.imshow(dilation)
plt.show()

#------------------------Opening and Closing------------------------------

opening=cv2.dilate(cv2.erode(img,se),se)
plt.imshow(opening)
plt.show()

closing=cv2.erode(cv2.dilate(img,se),se)
plt.imshow(closing)
plt.show()

#-----------------------------------------Closing and opening with morphologyEX--------------------------------
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,se) #یک سری از مورفولوژی ها رو در دل خودش دارد
plt.figure(2)
plt.imshow(closing)
plt.show()

opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,se) #یک سری از مورفولوژی ها رو در دل خودش دارد
plt.figure(2)
plt.imshow(opening)
plt.show()

