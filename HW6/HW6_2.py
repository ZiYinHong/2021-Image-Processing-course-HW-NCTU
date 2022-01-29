import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

#img = cv2.imread("Visual_resolution.tif")   #不可
# img = Image.open("Visual resolution.gif")  #可
img = mpimg.imread("Visual resolution.gif")  #可
print(img.shape)                             # (438, 612, 4)  # 最後一維是一條線
img  = img[:,:,:3]

plt.imshow(img), plt.title("img")
plt.show()

R = img[:,:,0]
plt.imshow(R, cmap="gray"), plt.title("R")
plt.show()

G = img[:,:,1]
plt.imshow(G, cmap="gray"), plt.title("G")
plt.show()

B = img[:,:,2]
plt.imshow(B, cmap="gray"), plt.title("B")
plt.show()
# =========================== Gradient computed in RGB color vector space ============================
# Sobelx
r_Rx = cv2.Sobel(R, cv2.CV_32F, 1, 0, ksize=3)  
g_Rx = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3)
b_Rx = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3)

# Sobely
r_Ry = cv2.Sobel(R, cv2.CV_32F, 0, 1, ksize=3)  
g_Ry = cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)
b_Ry = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3)

# gxx, gyy, gxy
gxx = r_Rx**2 + g_Rx**2 + b_Rx**2   # same as gxx = abs(r_Rx)**2 + abs(g_Rx)**2 + abs(b_Rx)**2 
gyy = r_Ry**2 + g_Ry**2 + b_Ry**2   # same as gyy = abs(r_Ry)**2 + abs(g_Ry)**2 + abs(b_Ry)**2
gxy = r_Rx*r_Ry + g_Rx*g_Ry + b_Rx*b_Ry

# angle(x, y)
print(f"(2*gxy).max(), (2*gxy).min() = {(2*gxy).max(), (2*gxy).min()}")
print(f"(gxx-gyy).max(), (gxx-gyy).min() = {(gxx-gyy).max(), (gxx-gyy).min()}")
angle = 1/2 * (np.arctan2((2*gxy),(gxx-gyy+1)))  # np.arctan2   # +1 prevent dividing zero 
angle = abs(angle)
print(f"angle.max(), angle.min() = {angle.max(), angle.min()}")    # (1.5707964, 0.0)
print(f"angle.shape = {angle.shape}")
"""
# np.arctan : The output of the function range from -180 to +180 
# np.arctan2 : The output of the function range from -90 to +90 
"""

# F(x, y)
F = (1/2* ( (gxx + gyy)+(gxx - gyy)*np.cos(2*angle) + 2*gxy*np.sin(2*angle) ) )**(1/2)
print(f"F.shape = {F.shape}")
plt.imshow(F , cmap="gray"), plt.title("F")
plt.show()

# ================= Sobel ========================================================================
## sobelx  
sobelRx = cv2.Sobel(R, cv2.CV_32F, 1, 0, ksize=3)  # x
sobelGx = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3)
sobelBx = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3)

## sobely
sobelRy = cv2.Sobel(R, cv2.CV_32F, 0, 1, ksize=3)  
sobelGy = cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)  
sobelBy = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3)  

sobel_total_R = np.abs(sobelRx) + np.abs(sobelRy)
sobel_total_G = np.abs(sobelGx) + np.abs(sobelGy)
sobel_total_B = np.abs(sobelBx) + np.abs(sobelBy)

plt.imshow(sobel_total_R, cmap="gray"), plt.title("sobel_total_R")
plt.show()
plt.imshow(sobel_total_G, cmap="gray"), plt.title("sobel_total_G")
plt.show()
plt.imshow(sobel_total_B, cmap="gray"), plt.title("sobel_total_B")
plt.show()

## sobel_total
sobel_total = sobel_total_R + sobel_total_G + sobel_total_B
plt.imshow(sobel_total, cmap="gray"), plt.title("sobel_total")
plt.show()


# ============================== Difference between F and sobel_total ===================================
Difference = F-sobel_total
print(f"Difference.max(), Difference.min() = {Difference.max(), Difference.min()}") 
plt.imshow( abs(Difference), cmap="gray"), plt.title("abs(F-sobel_total)")
plt.show()

#%%
import numpy as np
a = np.array([112, 300])
print(a)  # [112  44]


"""
https://stackoverflow.com/questions/13428689/whats-the-difference-between-cvtype-values-in-opencv

CV_8U - 8-bit unsigned integers ( 0..255 )

CV_8S - 8-bit signed integers ( -128..127 )

CV_16U - 16-bit unsigned integers ( 0..65535 )

CV_16S - 16-bit signed integers ( -32768..32767 )

CV_32S - 32-bit signed integers ( -2147483648..2147483647 )

CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )

CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )

8-bit unsigned integer (uchar)

8-bit signed integer (schar)

16-bit unsigned integer (ushort)

16-bit signed integer (short)

32-bit signed integer (int)

32-bit floating-point number (float)

64-bit floating-point number (double)
"""