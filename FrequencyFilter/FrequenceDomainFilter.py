import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('./images/img4.jpg',0)
img=(img-np.min(img))/(np.max(img)-np.min(img))
m = img.shape
c = np.floor(m[0]/2) #مرکزتصویر

ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
mag = np.log(1+np.abs(ft_shift))

nx, ny = (m[0], m[1])
x = np.linspace(-c, c, nx) #ساخت اعدادی از -127 تا 127 برای x , y که تعدادش 255
y = np.linspace(-c, c, ny)
xv, yv = np.meshgrid(x, y) #یک ماتریس داریم الان که هر عضو آن شامل (i,j)

r = xv**2 + yv**2 #فاصله همه تا مرکز

sigma = 100
# h = np.sqrt(r) <= sigma  #ideal lowpass filter
h = np.exp(-r/(2*sigma**2)) #gussian lowpass filter
# h = 1 - h #convert to highpass

y = ft_shift * h
mag2 = np.log(1+np.abs(y))

img2 = np.real(np.fft.ifft2(np.fft.ifftshift(y))) #عکس تبدیل فوریه

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(mag, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(h, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img2, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()