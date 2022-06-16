import cv2

img=cv2.imread("./images/img0.jpg")

print(img[0,0,0],img[0,0,1],img[0,0,2])
print(img[10,50,0],img[50,100,1],img[2,30,2])

img_grey=img[:,:,1]
print(img_grey[0,0])
img2=img[:,:,0:2]
print(img2)


img_slice=img[100:200,0:100,0] #patch_subimage
cv2.imshow("1",img_slice)
cv2.waitKey()
