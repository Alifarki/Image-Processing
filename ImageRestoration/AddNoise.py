import numpy as np
import matplotlib.pyplot as plt
import cv2

def noise_generator(noise_type, image): #ساخت نویز های ما
    #2 ورودی دارد که یکی نوع نویز و دیگری خود تصویر
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
        # ساخت نویز نمک 1
        num_salt = np.ceil(amount * image.size * s_vs_p) #سایز تصویر ضربدر میزان نویزamount ضربدر میزان نمک یا فلفل
        coords = [np.random.randint(0, i - 1, int(num_salt)) #ساخت یک سری اعداد تصادفی با سایز تصویر
                  for i in image.shape]
        out[coords] = 255
        # ساخت نویز فلفل 0
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



plt.figure(0)

img = cv2.imread('./images/img2.jpg', 0)
img = (img - np.min(img)) / (np.max(img) - np.min(img))

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
plt.close(0)


# plt.figure(1)
# sp_im = noise_generator('s&p', img)
# plt.imshow(sp_im, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.close(1)

plt.figure(2)
gauss_im = noise_generator('gauss', img)
plt.imshow(gauss_im, cmap='gray')
plt.title('Gaussian Noise')
plt.show()
plt.close(2)

# plt.figure(3)
# gauss_im = noise_generator('poisson', img)
# plt.imshow(gauss_im, cmap='gray')
# plt.title('poisson Noise')
# plt.show()
# plt.close(2)

# plt.figure(4)
# gauss_im = noise_generator('speckle', img)
# plt.imshow(gauss_im, cmap='gray')
# plt.title('speckle Noise')
# plt.show()
# plt.close(2)