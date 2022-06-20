import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('./images/img0.jpg',0)

img=(img-np.min(img))/(np.max(img)-np.min(img))
plt.imshow(img,cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.show()

ft=np.fft.fft2(img)
fshift=np.fft.fftshift(ft)
m_s=20*np.log(np.abs(fshift)) #magnitudeSpectrum #همون دامنه ماست.#DomainofImage
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(m_s, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

#------------------Phase-------------------
phase=np.angle(fshift) #محاسبه فاز
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(phase, cmap = 'gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.show()

#تا اینجا numpy بود و حالا همینا رو با cv2

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT) #خروجی را به صورت complex به من برگردان
dft_shift = np.fft.fftshift(dft) #شیفت دادن تصویر
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))#در صفحه سوم بعد سوم 0 و 1 دارند real  و imaginary را نمایش میدهند
phase = cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1])
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(phase, cmap = 'gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.show()

#fft2img
ft=np.fft.fft2(img)
fshift=np.fft.fftshift(ft)
m_s=20*np.log(np.abs(fshift))
img2=np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(m_s, cmap = 'gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2, cmap = 'gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.show()