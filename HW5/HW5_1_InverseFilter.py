import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy import signal
from DFT import DFT2d

def butterworth(img, D0, n, mode="lowpass"):
    # D0 is cut-off frequency, n is order, mode=lowpass/highpass
    M, N = img.shape
    x, y = np.mgrid[-M//2:M//2, -N//2:N//2]
    bw_filter = 1/(1+((x**2+y**2)/D0**2)**(n))   # 課本 p.282公式
    plt.plot(bw_filter.ravel()), plt.title("hist butterworth_filter (D0 = 85)")
    plt.show()
    if mode == "lowpass":
        return bw_filter 
    else:
        return (1 - bw_filter)
    

img = cv2.imread("Fig5.25.tif")
result = np.empty_like(img)

M, N, c  = img.shape
k = 0.0025

## degradation function H(u, v)
x, y = np.mgrid[0:M, 0:N]         #這樣定義x, y較符合直覺
H = np.exp(-k * (((x-M/2)**2 + (y-N/2)**2)**(5/6)) )


## 下面方式也行
# x, y = np.mgrid[-M//2:M//2, -N//2:N//2]
# H = np.exp(-k * (((x)**2 + (y)**2)**(5/6)) )


plt.imshow(np.log(abs(H) + 1), cmap='gray'), plt.title("H")
plt.show()


for i in range(c):
    g = img[:,:,i]
    # plt.imshow(g, cmap='gray'), plt.title("img")
    # plt.show()


    ## Fourier Transform of img
    #G = np.fft.fftshift(DFT2d(g))   # 因為原圖很大張 會算很慢
    G = np.fft.fftshift(np.fft.fft2(g))
    plt.imshow(np.log(abs(G) + 1), cmap='gray'), plt.title("G")
    plt.show()


    ## F = G/H
    F = G/H
    plt.imshow(np.log(abs(F) + 1), cmap='gray'), plt.title("F before filter")
    plt.show()

    ## ideal lowpass filter on F
    # r = 85
    # mask = np.zeros_like(g)
    # mask = cv2.circle(mask, (N//2, M//2), r, (1,1,1), -1) 
    # F = F*mask

    ## butterworth filter
    D0 = 85
    n = 25
    bw_filter = butterworth(F, D0, n, mode="lowpass")
    F = F*bw_filter


    plt.imshow(np.log(abs(F) + 1), cmap='gray'), plt.title("F after filter(D0 = 85)")
    plt.show()


    ## inverse Fourier Transform of F
    ifft_img = np.fft.ifft2(np.fft.ifftshift(F))
    ifft_img = np.abs(ifft_img)
    plt.imshow(ifft_img, cmap='gray'), plt.title("ifft_img")
    plt.show()


    ifft_img_n = (ifft_img - ifft_img.min())/(ifft_img.max() - ifft_img.min())*255
    result[:,:,i] = ifft_img_n

plt.imshow(result, cmap='gray'), plt.title("result")
plt.show()
