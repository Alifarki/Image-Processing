import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
import cv2

# astro = color.rgb2gray(data.astronaut())

# psf = np.ones((5, 5)) / 25
# astro = conv2(astro, psf, 'same')
# astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)
#برای فیلتر وینر نتیجه بهتر در صورتی است که نویز هم اعمال شده باشد.

astro = cv2.imread('./images/img2.jpg', 0)
astro = (astro - np.min(astro)) / (np.max(astro) - np.min(astro))
psf = np.ones((5, 5)) / 25 #فیلتر ایجاد تحرک و حرکت در تصویر
astro = cv2.filter2D(astro, -1, psf)
gauss = np.random.normal(0, 0.001, astro.shape) #اعمال نویز
astro += gauss
#wiener
deconvolved, _ = restoration.unsupervised_wiener(astro, psf) #فیلتر وینر
#مقدار تخریب مثلا تکان خوردن دست هنگام عکاسی را باید حساب بکنیم.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()
#astro=degrade image
#deconvolved=image after restore
ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()