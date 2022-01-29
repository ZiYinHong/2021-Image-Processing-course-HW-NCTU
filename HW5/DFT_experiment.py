import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy import signal
import time

def DFT2d(f): 
    """
     f is a two dimansional, one channel image
    """
    M, N  = f.shape
    F = np.zeros((M,N),dtype=complex)   # so important!!! datatype must be in complex

    ## return DFT image F(u, v)   
    for u in range(0, M):
        for v in range(0, N):
            print(f"u = {u}, v = {v}")
            for x in range(0, M):
                for y in range(0, N):
                    F[u, v] += f[x, y]*np.exp(-2j* np.pi* (u*x/M + v*y/N))
    return F


img = cv2.imread("Fig5.25.tif")
g = img[:,:, 0]
g = cv2.resize(g, (32, 32))
plt.imshow(g, cmap='gray'), plt.title("g")
plt.show()

## Fourier Transform of img by hand
# padding g first
# pad_g = np.zeros((64, 64))
# pad_g[:32, :32] = g
# G = DFT(pad_g)

start = time.time()
G = DFT2d(g)
end = time.time()
print("執行時間：%f 秒" % (end - start))   # 執行時間：5.740395 秒

plt.imshow(np.log(abs(G) + 1), cmap='gray'), plt.title("g after DFT")        
plt.show()
G = np.fft.fftshift(G)          # P.242
plt.imshow(np.log(abs(G) + 1), cmap='gray'), plt.title("G after fftshift")       
plt.show()

## ====================== conclusion ==============================================
# dtype 指定為 complex 重要
# 不用先 padding

## Fourier Transform of img, using Library
start = time.time()
G = np.fft.fftshift(np.fft.fft2(g))
end = time.time()
print("執行時間：%f 秒" % (end - start))     # 執行時間：0.000169 秒   跟自己寫的差很多 

plt.imshow(np.log(abs(G) + 1), cmap='gray'), plt.title("G")
plt.show()