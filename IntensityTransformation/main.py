import cv2
import numpy as np
img=cv2.imread("./images/img0.jpg")
cv2.imshow("0",img)
cv2.waitKey()
#-------------------------------------------------
img=img[:,:,0] #img0 has 3 channels so we must change it to one channel.(convert to 1 channel)
#-------------------------------------------------

m=img.shape
num=2
t=np.linspace(0,255,num,dtype=int)
img_new=np.zeros(img.shape,np.uint8) #create numpy array that has shape of img
for i in range(m[0]):
    for j in range(m[1]): #for every value of i we have all values of j  (row based move)
        for k in range(len(t)-1):
            if img[i][j] >= t[k] and img[i][j] < t[k + 1]:
                middle=((t[k]+t[k+1])/2)
                img[i][j] = t[k]
cv2.imshow("Black&White",img)
cv2.waitKey()

