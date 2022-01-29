import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("fish.jpg") 
img = img[:,:,::-1]
plt.imshow(img), plt.title("original img")
plt.show()
print(img.shape)

### origin highboost
blur = cv2.GaussianBlur(img, (31, 31), 5, cv2.BORDER_REFLECT)

mask = img - blur
print(mask.max(), mask.min())
plt.imshow(mask), plt.title("mask")
plt.show()

unsharp = img + mask
plt.imshow(unsharp), plt.title("unsharp")
plt.show()

k = 2
highboost = img + k*mask
plt.imshow(highboost), plt.title("original highboost")
plt.show()

plt.subplot(121), plt.imshow(img), plt.axis("off"), plt.title("original img")
plt.subplot(122), plt.imshow(highboost), plt.axis("off"), plt.title("original highboost")
plt.show()



#=========================================================================================================
### Laplacian + highboost (p.193 FIGURE 3.57 (c))
# img = cv2.imread("fish.jpg") 
# print(img.shape)
# img = img[:,:,::-1]
# plt.imshow(img), plt.title("original img")
# plt.show()

## use self-designed laplacian kernel
laplacian_kernel = np.array([[-1,-1,-1],
                            [-1, 8,-1],
                            [-1,-1,-1]])
                            
laplacian2 = cv2.filter2D(img, cv2.CV_16S, laplacian_kernel)  # ddepth = -1
print(f"laplacian2.max(), laplacian2.min() = {laplacian2.max(), laplacian2.min()}") 
laplacian2_n = ((laplacian2 - laplacian2.min())/ (laplacian2.max() - laplacian2.min()))*255  # for display
# plt.imshow(laplacian2_n.astype(np.uint8)), plt.title("laplacian2_n")
# plt.show()


unsharp2 = img + laplacian2
print(f"unsharp2.max(), unsharp2.min() = {unsharp2.max(), unsharp2.min()}") 
# plt.imshow(unsharp2), plt.title("laplacian2 unsharp2")  # best
# plt.show()
unsharp2_n = ( (unsharp2 - unsharp2.min()) / (unsharp2.max() - unsharp2.min()))*255
# plt.imshow(unsharp2_n.astype(np.uint8)), plt.title("laplacian2 unsharp2_n")
# plt.show()


# plt.subplot(121), plt.imshow(img), plt.axis("off"), plt.title("original img")
# plt.subplot(122), plt.imshow(unsharp2), plt.axis("off"), plt.title("laplacian unsharp2")
# plt.show()


#=========================================================================================================
### Sobel + highboost (p.193 FIGURE 3.57 (c))
img = cv2.imread("fish.jpg") 
print(img.shape)
img = img[:,:,::-1]
plt.imshow(img), plt.title("original img")
plt.show()

## sobelx  
sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)  # x
print(f"sobelx.max(), sobelx.min() = {sobelx.max(), sobelx.min()}") 
sobelx2 = np.abs(sobelx).astype(np.uint8)
# plt.imshow(sobelx2), plt.title("sobelx2")
# plt.show()

## sobely
sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)  
print(f"sobely.max(), sobely.min() = {sobely.max(), sobely.min()}") 
sobely2 = np.abs(sobely).astype(np.uint8)
# plt.imshow(sobely2), plt.title("sobely2")
# plt.show()

sobel_total = np.abs(sobelx) + np.abs(sobely)
print(f"sobel_total.max(), sobel_total.min() = {sobel_total.max(), sobel_total.min()}") 
plt.imshow(sobel_total), plt.title("sobel_total")
plt.show()


# Sobel image smoothed with a 3×3 box filter.
box_filter = np.ones((3,3))*(1/9)
Sobel_blur = cv2.filter2D(sobel_total, cv2.CV_16S, box_filter)
print(f"Sobel_blur.max(), Sobel_blur.min() = {Sobel_blur.max(), Sobel_blur.min()}")
plt.imshow(Sobel_blur), plt.title("Sobel_blur")
plt.show()

# Mask image formed by the product of Laplacian and Sobel_blur.
mask = Sobel_blur * laplacian2
print(f"mask.max(), mask.min() = {mask.max(), mask.min()}")
mask[mask<0] = 0
# plt.imshow(mask), plt.title("mask")
# plt.show()

mask2 = (((mask - mask.min()) / (mask.max() - mask.min()))*255).astype("uint8")
print(f"mask2.max(), mask2.min() = {mask2.max(), mask2.min()}")
# plt.imshow(mask2), plt.title("mask2")
# plt.show()


# Sharpened image obtained by the adding images img and mask2.
Sharpened = img.astype("uint16") + mask2.astype("uint16")
print(f"Sharpened.max(), Sharpened.min() = {Sharpened.max(), Sharpened.min()}")

plt.imshow(Sharpened), plt.title("Sharpened")
plt.show()
plt.imshow(highboost + Sharpened), plt.title("highboost + Sharpened")
plt.show()


plt.subplot(411), plt.imshow(img), plt.axis("off"), plt.title("original img")
plt.subplot(412),plt.imshow(highboost), plt.axis("off"), plt.title("highboost")
plt.subplot(413),plt.imshow(Sharpened), plt.axis("off"), plt.title("Sharpened")
plt.subplot(414),plt.imshow(highboost + Sharpened), plt.axis("off"), plt.title("highboost + Sharpened")
plt.show()






