import cv2
img=cv2.imread("./images/img2.jpg")
img=img[:,:,0]
imgd=cv2.normalize(img.astype('float'),None,0.0,1.0,cv2.NORM_MINMAX)

img_pow=imgd ** 2
cv2.imshow("pow",img_pow)
cv2.waitKey(0)

img_pow2=imgd ** 0.2
cv2.imshow("pow2",img_pow2)
cv2.waitKey(0)