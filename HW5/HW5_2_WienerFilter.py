import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy import signal
from scipy.signal.signaltools import wiener


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
# plt.imshow(np.log(abs(H) + 1), cmap='gray'), plt.title("H")
# plt.show()


for i in range(c):
    g = img[:,:,i]


    ## Fourier Transform of img
    G = np.fft.fftshift(np.fft.fft2(g))


    ## wiener filter
    # for p in np.arange(1, 20, 1):
    #     print(p)
    p = 4  # p=4 is the best
    K = 1/10**p  
    W = (1/H)*(abs(H)**2/(abs(H)**2+K))
    F = W*G
    # plt.imshow(np.log(abs(F) + 1), cmap='gray'), plt.title("F")
    # plt.show()


    ## Inverse Fourier Transform
    ifft_F = np.fft.ifft2(np.fft.ifftshift(F))
    ifft_F = np.abs(ifft_F)
    # plt.imshow(ifft_F, cmap='gray'), plt.title("ifft_F")
    # plt.show()

    ifft_F_n = (ifft_F - ifft_F.min())/(ifft_F.max() - ifft_F.min())*255
    result[:,:,i] = ifft_F_n

plt.imshow(result, cmap='gray'), plt.title("result")
plt.show()

