import cv2
img=cv2.imread("./images/img2.jpg")
img=img[:,:,0]
imgd=cv2.normalize(img.astype('float'),None,0.0,1.0,cv2.NORM_MINMAX)
img_neg=1-imgd
cv2.imshow("negative",img_neg)
cv2.waitKey(0)