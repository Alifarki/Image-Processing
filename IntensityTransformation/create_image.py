import cv2
img=cv2.imread("./images/img2.jpg")
cv2.imshow('first',img)
cv2.waitKey()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('sec',gray)
cv2.waitKey()
# import numpy as np
# import copy
# img=np.zeros((500,500,3),np.uint8)
# cv2.imshow('first',img)
# cv2.waitKey()
# img[100:200,200:300,0]=255 #in opencv first channel=blue,second=green and thirs is red BGR
# # cv2.cvtColor(cv2.COLOR_RGB2BGR) #convert to another channel and grey scale
# #it show a blue square
# cv2.imshow('second',img)
# cv2.waitKey()
#
# tmp=copy.deepcopy(img[:,:,0]) #B #chane BGR to another B or G or R
# img[:,:,0]=copy.deepcopy(img[:,:,2])
# img[:,:,2]=copy.deepcopy(tmp)
# cv2.imshow('0',img)
# cv2.waitKey()
# #Circle-How
# # x=x0+r*cos(theta), y=y0+r*sin(theta)
