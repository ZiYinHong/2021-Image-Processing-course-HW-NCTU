import cv2

# importing PIL
from PIL import Image

# importing matplotlib modules
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

"""
# opencv read image
img = cv2.imread("astronaut-interference.tif", 0)   # read in gray level
print(img.shape)
cv2.imshow("img", img)
cv2.waitKey()
"""

# matplotlib read image
img = mpimg.imread("astronaut-interference.tif")
print(img.shape)
# Output Images
plt.imshow(img, cmap=cm.gray)
plt.show()


"""
# PIL read image
img = Image.open("checkerboard1024-shaded.tif")
# Output Images
img.show()
# img = cv2.imread("checkerboard1024-shaded.tif")
print(img.shape)
"""

# Fourier Transform
fft_img = np.fft.fftshift(np.fft.fft2(img))
plt.imshow(np.log(abs(fft_img) + 1), cmap='gray'), plt.title("fft_img")
plt.show()
# 實驗 ===================================================================
# value = np.log(abs(fft_img) + 1)
# test_img = value/value.max()
# print(test_img.max(), test_img.min())
# plt.imshow(test_img , cmap='gray'), plt.title("test_img")
# plt.show()
# ===================================================================



## modify on FFT image
mask = np.zeros_like(img)
mask = cv2.circle(mask, (475,386), 2, (255,255,255), -1)  # 左上角 burst
mask = cv2.circle(mask, (525,436), 2, (255,255,255), -1)  # 右下角 burst
mask = 255 - mask
mask = mask/mask.max()
plt.imshow(mask, cmap='gray'), plt.title("mask")
plt.show()


modify_fft_img = fft_img*mask
plt.imshow(np.log(abs(modify_fft_img) + 1), cmap='gray'), plt.title("modify fft_img")
plt.show()


## inverse Fourier Transform
ifft_img = np.fft.ifft2(np.fft.ifftshift(modify_fft_img))
print(f"ifft_img.max(), ifft_img.min() = {ifft_img.max(), ifft_img.min()}")
plt.imshow(np.abs(ifft_img), cmap='gray'), plt.title("ifft_img")
plt.show()

#%%
import numpy as np
a = [1+2j, 4+5j]
print(np.abs(a))
print(np.sqrt(5))
