import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

"""
img = cv2.imread("Bodybone.bmp") 
img = img[:,:,::-1]
plt.imshow(img)
plt.show()
#print(img.shape)

### origin highboost
blur = cv2.GaussianBlur(img, (31, 31), 5, cv2.BORDER_REFLECT)

mask = img - blur
print(mask.max(), mask.min())
plt.imshow(mask), plt.title("mask")
plt.show()

unsharp = img + mask
plt.imshow(unsharp), plt.title("unsharp")
plt.show()

k = 3
highboost = img + k*mask
plt.imshow(highboost), plt.title("highboost")
plt.show()

plt.subplot(131), plt.imshow(img), plt.axis("off"), plt.title("original img")
plt.subplot(132), plt.imshow(unsharp), plt.axis("off"), plt.title("unsharp")
plt.subplot(133), plt.imshow(highboost), plt.axis("off"), plt.title("highboost")
plt.show()
"""

#===================================================================================================
### Laplacian  (p.193 FIGURE 3.57 (b) (c))
img = cv2.imread("Body.tif") 
print(f"img.dtype = {img.dtype}")
print(img.shape)
img = img[:,:,::-1]
plt.imshow(img), plt.title("original img")
plt.show()

## use self-defined laplacian kernel
laplacian_kernel = np.array([[-1,-1,-1],
                            [-1, 8,-1],
                            [-1,-1,-1]])
                            
laplacian = cv2.filter2D(img, cv2.CV_16S, laplacian_kernel)  # ddepth cv2.CV_16S : 16-bit signed integers
# plt.imshow(laplacian), plt.title("laplacian")
# plt.show()
print(f"laplacian.max(), laplacian.min() = {laplacian.max(), laplacian.min()}") 

# for display
laplacian_n = (((laplacian - laplacian.min())/ (laplacian.max() - laplacian.min()))*255).astype(np.uint8)  
print(f"laplacian_n.max(), laplacian_n.min() = {laplacian_n.max(), laplacian_n.min()}") 
plt.imshow(laplacian_n), plt.title("laplacian_n")
plt.show()


#===================================================================================================
### unsharp image by img + laplacian (c)
unsharp = img + laplacian
print(f"unsharp.max(), unsharp.min() = {unsharp.max(), unsharp.min()}") 
plt.imshow(unsharp), plt.title("unsharp")  # best
plt.show()
# unsharp_n = ( (unsharp - unsharp.min()) / (unsharp.max() - unsharp.min()))*255
# plt.imshow(unsharp_n.astype(np.uint8)), plt.title("unsharp_n")
# plt.show()


plt.subplot(121), plt.imshow(img), plt.axis("off"), plt.title("original img")
plt.subplot(122), plt.imshow(unsharp), plt.axis("off"), plt.title("laplacian unsharp")
plt.show()

#===================================================================================================
### Sobel  (p.193 FIGURE 3.57 (d))
img = cv2.imread("Body.tif")
img = img[:,:,::-1] 

## sobelx  
sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)  # x
print(f"sobelx.max(), sobelx.min() = {sobelx.max(), sobelx.min()}") 

sobelx2 = np.abs(sobelx).astype(np.uint8)
sobelx_n = cv2.normalize(src=np.abs(sobelx), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# plt.imshow(sobelx2), plt.title("sobelx2")
# plt.show()
# plt.imshow(sobelx_n), plt.title("sobelx_n")
# plt.show()


## sobely
sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)  
print(f"sobely.max(), sobely.min() = {sobely.max(), sobely.min()}") 

sobely2 = np.abs(sobely).astype(np.uint8)

sobely_n = cv2.normalize(src=np.abs(sobely), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# plt.imshow(sobely2), plt.title("sobely2")
# plt.show()
# plt.imshow(sobely_n), plt.title("sobely_n")
# plt.show()


sobel_total = np.abs(sobelx) + np.abs(sobely)
#print(sobel_total.dtype)  #int16
print(f"sobel_total.max(), sobel_total.min() = {sobel_total.max(), sobel_total.min()}") 
plt.imshow(sobel_total), plt.title("sobel_total")
plt.show()


"""
## unsharp
unsharp = img + sobel_total
print(f"unsharp.max(), unsharp.min() = {unsharp.max(), unsharp.min()}") 
plt.imshow(unsharp), plt.title("sobel unsharp")
plt.show()
unsharp_n = ( (unsharp - unsharp.min()) / (unsharp.max() - unsharp.min()))*255
plt.imshow(unsharp_n.astype(np.uint8)), plt.title("unsharp_n")
plt.show()

sobel_unsharp = unsharp

plt.subplot(131), plt.imshow(img), plt.axis("off"), plt.title("original img")
plt.subplot(132), plt.imshow(laplacian_unsharp), plt.axis("off"), plt.title("laplacian unsharp")
plt.subplot(133), plt.imshow(sobel_unsharp), plt.axis("off"), plt.title("sobel unsharp")
plt.show()
"""

#===================================================================================================
# Sobel image smoothed with a 5×5 box filter. (e)
box_filter = np.ones((5,5))*(1/25)
Sobel_blur = cv2.filter2D(sobel_total, cv2.CV_16S, box_filter)
print(f"Sobel_blur.max(), Sobel_blur.min() = {Sobel_blur.max(), Sobel_blur.min()}")
plt.imshow(Sobel_blur), plt.title("Sobel_blur")
plt.show()


#===================================================================================================
# Mask image formed by the product of Laplacian and Sobel_blur. (f)
mask = Sobel_blur * laplacian
#print(mask.dtype)  #int16
print(f"mask.max(), mask.min() = {mask.max(), mask.min()}")
mask[mask<0] = 0

mask_n = (((mask - mask.min()) / (mask.max() - mask.min()))*255).astype("uint8")
print(f"mask_n.max(), mask_n.min() = {mask_n.max(), mask_n.min()}")
# mask_n[mask_n>1] =  mask_n[mask_n>1] + 70
plt.imshow(mask_n), plt.title("mask_n")
plt.show()


#===================================================================================================
# unsharp2 image obtained by the adding images img and mask_n. (g)
unsharp2 = img.astype("uint16") + mask_n.astype("uint16")
print(f"unsharp2.max(), unsharp2.min() = {unsharp2.max(), unsharp2.min()}")
# hist = cv2.calcHist([unsharp2], [0], None, [484], [0, 484])
# plt.plot(hist)
# plt.show()
#print(unsharp2.dtype)  #uint8

# unsharp2 = (((unsharp2 - unsharp2.min()) / (unsharp2.max() - unsharp2.min()))*255).astype("uint8")
# hist = cv2.calcHist([unsharp2], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()

plt.imshow(unsharp2), plt.title("unsharp2")
plt.show()


#===================================================================================================
# Final result obtained by applying a powerlaw transformation (h)
c = 1
gamma = 0.5
powerlaw = unsharp2**gamma*c
print(f"powerlaw.max(), powerlaw.min() = {powerlaw.max(), powerlaw.min()}")

powerlaw = (((powerlaw - powerlaw.min()) / (powerlaw.max() - powerlaw.min()))*255).astype("uint8")
plt.imshow(powerlaw), plt.title("powerlaw")
plt.show()


#===================================================================================================
# show all results at once
# 課本 p.195~p.197
plt.subplot(241), plt.imshow(img), plt.axis("off"), plt.title("original img (a)")
plt.subplot(242), plt.imshow(laplacian_n), plt.axis("off"), plt.title("Laplacian of img (b)")
plt.subplot(243), plt.imshow(unsharp), plt.axis("off"), plt.title("unsharp img (c)")
plt.subplot(244), plt.imshow(sobel_total), plt.axis("off"), plt.title("Sobel img (d)")
plt.subplot(245), plt.imshow(Sobel_blur), plt.axis("off"), plt.title("blur Sobel img (e)")
plt.subplot(246), plt.imshow(mask_n), plt.axis("off"), plt.title("mask img (f)")
plt.subplot(247), plt.imshow(unsharp2), plt.axis("off"), plt.title("unsharp2 img2 (g)")
plt.subplot(248), plt.imshow(powerlaw), plt.axis("off"), plt.title("powerlaw transformation img (h)")
plt.show()


#%%
import cv2
import numpy as np
A = np.array([2, 200, 2])
B = np.array([3, 255, 8])
C = cv2.bitwise_and(A , B)
D = A + B
E = cv2.add(A, B)
print(C)
print(D)
print(E)


