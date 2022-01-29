import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from numpy.lib import math
from scipy import signal
import cmath

img = cv2.imread("book-cover-blurred.tif")
result = np.empty_like(img)
M, N, c  = img.shape
a = 0.1
b = 0.1
T = 1

## degradation function H(u, v)
x, y = np.mgrid[-M//2:M//2, -N//2:N//2]
k = np.pi*(x*a+y*b)
H = (T/(k)*np.sin(k))*np.exp(k*-1j)
H[np.isnan(H)] = T

#H = np.fft.fftshift(H)
plt.imshow(np.log(abs(H) + 1), cmap='gray'), plt.title("H")
plt.show()


for i in range(c):
    g = img[:,:,i]
    # plt.imshow(g, cmap='gray'), plt.title("img")
    # plt.show()


    ## Fourier Transform of img
    G = np.fft.fftshift(np.fft.fft2(g))
    # plt.imshow(np.log(abs(G) + 1), cmap='gray'), plt.title("fft_img")
    # plt.show()

    ## F = G/H
    F = G/H

    ## inverse Fourier Transform of F
    ifft_img = np.fft.ifft2(np.fft.ifftshift(F))
    ifft_img = np.abs(ifft_img)
    # plt.imshow(ifft_img, cmap='gray'), plt.title("ifft_img")
    # plt.show()


    ifft_img_n = (ifft_img - ifft_img.min())/(ifft_img.max() - ifft_img.min())*255
    result[:,:,i] = ifft_img_n

plt.imshow(result, cmap='gray'), plt.title("result")
plt.show()

