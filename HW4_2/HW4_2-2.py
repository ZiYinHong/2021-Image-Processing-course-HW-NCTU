import cv2

# importing PIL
from PIL import Image

# importing matplotlib modules
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import signal


def butterFilter(order, CutOffFrequency, mask_img, row1, row2, col1, col2):
    # ValueError: Digital filter critical frequencies must be 0 < Wn < 1
    img = mask_img[row1:row2, col1:col2]
    b, a = signal.butter(order, CutOffFrequency, btype='lowpass')  
    bf_img = signal.filtfilt(b, a, img)
    return bf_img

# matplotlib read image
img = mpimg.imread("car-moire-pattern.tif")
print(img.shape)
# Output Images
plt.imshow(img, cmap=cm.gray)
plt.show()


# Fourier Transform
fft_img = np.fft.fftshift(np.fft.fft2(img))
plt.imshow(np.log(abs(fft_img) + 1), cmap='gray'), plt.title("fft_img")
plt.show()


## modify on FFT image
mask = np.zeros_like(img)
# 左邊那排
mask = cv2.circle(mask, (54,42), 10, (255,255,255), -1)  
mask = cv2.circle(mask, (54,84), 10, (255,255,255), -1)  
mask = cv2.circle(mask, (56,165), 10, (255,255,255), -1)
mask = cv2.circle(mask, (56,205), 10, (255,255,255), -1)
# 右邊那排
mask = cv2.circle(mask, (110,40), 10, (255,255,255), -1)  
mask = cv2.circle(mask, (110,80), 10, (255,255,255), -1)  
mask = cv2.circle(mask, (113,161), 10, (255,255,255), -1)
mask = cv2.circle(mask, (113,202), 10, (255,255,255), -1)
# 實驗 mask掉其他斜線=======================================================
# 右斜
# contours = np.array([[59,0], [61,0], [107,246], [105,246]])  # [x, y] reference: https://stackoverflow.com/questions/11270250/what-does-the-python-interface-to-opencv2-fillpoly-want-as-input
# cv2.fillPoly(mask, pts = [contours], color =(255,255,255))
# #正中
# mask[:, 83:86] = 255
# 左右兩條斜線
# mask[:, 53:57] = 255  # 左
# mask[:, 109:113] = 255  # 右

# 結論 :
# mask掉正中或右斜 會像highpass filter
# mask掉左右兩條斜線 沒差

mask = 255 - mask
mask = mask/mask.max()
plt.imshow(mask, cmap='gray'), plt.title("mask")
plt.show()


# 實驗 butterworth ================================================================
# bf_img = butterFilter(4, 0.4, mask, 32, 54, 44, 66)
# mask[32:54, 44:66] = bf_img
# bf_img = butterFilter(4, 0.4, mask, 74, 96, 44, 66)
# mask[74:96, 44:66] = bf_img
# bf_img = butterFilter(4, 0.4, mask, 155, 177, 46, 68)
# mask[155:177, 46:68] = bf_img
# bf_img = butterFilter(4, 0.4, mask, 195, 217, 46, 68)
# mask[195:217, 46:68] = bf_img
# bf_img = butterFilter(4, 0.4, mask, 32, 52, 100, 122)
# mask[32:52, 100:122] = bf_img
# bf_img = butterFilter(4, 0.4, mask, 70, 92, 100, 122)
# mask[70:92, 100:122] = bf_img
# bf_img = butterFilter(4, 0.4, mask, 151, 173, 103, 125)
# mask[151:173, 103:125] = bf_img
# bf_img = butterFilter(4, 0.4, mask, 192, 214, 103, 125)
# mask[192:214, 103:125] = bf_img
# mask_n = (mask-mask.min())/(mask.max() - mask.min())
# print(mask.max(), mask.min())
# plt.imshow(mask, cmap='gray'), plt.title("mask_butterworth")
# plt.show()
# cv2.imwrite("mask_butterworth.png", mask_n*255)
#結論 :紋路部分好像變模糊了一點，但沒差很多


# 實驗 Gaussian filter ================================================
# mask[32:54, 44:66] = cv2.GaussianBlur(mask[32:54, 44:66], (7, 7), 5, cv2.BORDER_REFLECT)
# mask[74:96, 44:66] = cv2.GaussianBlur(mask[74:96, 44:66], (7, 7), 5, cv2.BORDER_REFLECT)
# mask[155:177, 46:68] = cv2.GaussianBlur(mask[155:177, 46:68], (7, 7), 5, cv2.BORDER_REFLECT)
# mask[195:217, 46:68] = cv2.GaussianBlur(mask[195:217, 46:68], (7, 7), 5, cv2.BORDER_REFLECT)
# mask[32:52, 100:122] = cv2.GaussianBlur(mask[32:52, 100:122], (7, 7), 5, cv2.BORDER_REFLECT)
# mask[70:92, 100:122] = cv2.GaussianBlur(mask[70:92, 100:122], (7, 7), 5, cv2.BORDER_REFLECT)
# mask[151:173, 103:125] = cv2.GaussianBlur(mask[151:173, 103:125], (7, 7), 5, cv2.BORDER_REFLECT)
# mask[192:214, 103:125] = cv2.GaussianBlur(mask[192:214, 103:125], (7, 7), 5, cv2.BORDER_REFLECT)
# plt.imshow(mask, cmap='gray'), plt.title("mask_gaussian")
# plt.show()
##結論 :紋路部分好像變模糊了一點，感覺沒差很多

modify_fft_img = fft_img*mask
plt.imshow(np.log(abs(modify_fft_img) + 1), cmap='gray'), plt.title("modify fft_img")
plt.show()



## inverse Fourier Transform
ifft_img = np.fft.ifft2(np.fft.ifftshift(modify_fft_img))
print(f"ifft_img.max(), ifft_img.min() = {ifft_img.max(), ifft_img.min()}")
plt.imshow(np.abs(ifft_img), cmap='gray'), plt.title("ifft_img")
plt.show()



