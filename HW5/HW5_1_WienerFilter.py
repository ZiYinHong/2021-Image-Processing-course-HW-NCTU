import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy import signal
from scipy.signal.signaltools import wiener


img = cv2.imread("Fig5.25.tif")
result = np.empty_like(img)

M, N, c  = img.shape
k = 0.0025

## degradation function H(u, v)
x, y = np.mgrid[-M//2:M//2, -N//2:N//2]
H = np.exp(-k * (((x)**2 + (y)**2)**(5/6)) )


for i in range(c):
    g = img[:,:,i]

    ## Fourier Transform of img
    G = np.fft.fftshift(np.fft.fft2(g))

    ## calculate K  !!!doesn't work
    # ifft_H = np.fft.ifft2(np.fft.ifftshift(H))
    # mean = np.mean(ifft_H)                          # calculate noise's mean
    # variance = np.sum((ifft_H - mean)**2)/(M*N)     # calculate noise's variance
    # noise_power_spectrum =  M*N*(variance + mean**2)
    # degraded_img_spectrum = np.sum(abs(G)**2)
    # K = noise_power_spectrum/degraded_img_spectrum
    # print(f"K = {K}")


    ## wiener filter
    K = 0.0001
    W = (1/H)*(abs(H)**2/(abs(H)**2+K))
    F = W*G
    # plt.imshow(np.log(abs(F) + 1), cmap='gray'), plt.title("F")
    # plt.show()

    ## Inverse Fourier Transform
    ifft_F = np.fft.ifft2(np.fft.ifftshift(F))
    ifft_F = np.abs(ifft_F)
    # plt.imshow(ifft_F, cmap='gray'), plt.title("ifft_F")
    # plt.show()


    result[:,:,i] = ifft_F

plt.imshow(result, cmap='gray'), plt.title("result")
plt.show()
