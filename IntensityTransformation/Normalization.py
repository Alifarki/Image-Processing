import cv2
import numpy as np
img=cv2.imread("./images/img2.jpg")
img=img[:,:,0]

img_norm01=((img - np.min(img))/(np.max(img)-np.min(img)))
cv2.imshow("1",img_norm01)
cv2.waitKey(0)


img_norm0255=np.uint8(255 * ((img_norm01 - np.min(img_norm01))/(np.max(img_norm01)-np.min(img_norm01))))
cv2.imshow("2",img_norm0255)
cv2.waitKey(0)

img_norm60188=np.uint8(128 * ((img_norm01 - np.min(img_norm01))/(np.max(img_norm01)-np.min(img_norm01)))+60)
cv2.imshow("3",img_norm60188)
cv2.waitKey(0)

img1=img.astype('float')
img_normCV=cv2.normalize(img1,None,0.0,1.0,cv2.NORM_MINMAX)
cv2.imshow("4",img_normCV)
cv2.waitKey(0)