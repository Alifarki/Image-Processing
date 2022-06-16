import cv2
import numpy as np
img=cv2.imread("./images/img0.jpg")
cv2.imshow("0",img)
cv2.waitKey()
#-------------------------------------------------
img=img[:,:,0] #img0 has 3 channels so we must change it to one channel.(convert to 1 channel)
#-------------------------------------------------

# print(img.size)
# print(img.shape)

m=img.shape
t=[50,178]
img_g=np.zeros(img.shape,np.uint8) #create numpy array that has shape of img
for i in range(m[0]):
    for j in range(m[1]): #for every value of i we have all values of j  (row based move)
        if img[i,j] <= 50:
            img_g[i,j]=1000
        elif img[i,j] > 50 and img[i,j]<=178:
            img_g[i,j]=1
        else:
            img_g[i,j]=2
cv2.imshow("Black&White",img_g)
cv2.waitKey()
print(img_g)

#-------------------------------------------------
