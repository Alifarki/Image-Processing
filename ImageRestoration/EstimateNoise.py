import numpy as np
import matplotlib.pyplot as plt
import cv2

def noise_generator(noise_type, image):
    row, col = image.shape
    if noise_type == "gauss":
        mean = 0.0
        sigma = 0.1
        gauss = np.random.normal(mean, sigma, img.shape)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy
    else:
        return image
def plot_hist(img):
    img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img.astype(np.uint8)
    histr = np.zeros(256) #محاسبه هیستوگرام طبق رسم هیستوگرام
    for i in range(256):
        idx = np.sum(img == i)
        histr[i] = idx
    return histr

img = cv2.imread('./images/moon.tif', 0)
img = (img - np.min(img)) / (np.max(img) - np.min(img))

plt.figure(0)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
plt.close(0)

plt.figure(2)
gauss_im = noise_generator('gauss', img)
plt.imshow(gauss_im, cmap='gray')
plt.title('Gaussian Noise')
plt.show()
plt.close(2)


plt.figure(3)
img_roi = gauss_im[20:100,20:100]
plt.imshow(img_roi, cmap='gray')
plt.title('ROI') #regionofinterest
plt.show()
plt.close(2)

plt.figure(3)
histr = plot_hist(img_roi)  #هیستوگرام همون roi ماست
plt.bar(np.arange(256), histr, color = 'b')
plt.show()
#نویز گوسین اضافه شد و الان با استفاده از دیدن هیستوگرام میتوانیم نوع نویز را بفهمیم