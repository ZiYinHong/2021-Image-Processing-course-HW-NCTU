import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

img = cv2.imread("checkerboard1024-shaded.tif") 
print(img.max())
## check 是否三層都相同
# print((img[:,:,0] == img[:,:,1]).all() == True)  # True
# print((img[:,:,0] == img[:,:,2]).all() == True)  # True
# print((img[:,:,1] == img[:,:,2]).all() == True)  # True

blur = cv2.GaussianBlur(img, (255, 255), 64, cv2.BORDER_REFLECT)
print(blur.max())

result = img / blur
print(f"result.max(), result.min() = {result.max(), result.min()}")
#result_n = ( (result - result.min()) / (result.max() - result.min()))*255
cv2.imwrite("result.png", (result*255).astype("uint8"))
cv2.imshow("result", result)
cv2.waitKey()

# cv2.imshow("img", img)
# cv2.imshow("blur", blur)
# cv2.imshow("result ", result)
# cv2.waitKey()

## two way doing subplot in matplotlib
# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(img)
# ax[1].imshow(blur)
# ax[2].imshow(result)

plt.subplot(131),plt.imshow(img)
plt.subplot(132),plt.imshow(blur)
plt.subplot(133),plt.imshow(result)
plt.show()
plt.tight_layout()

#===========================================================================
### 手刻 Gassian filter
#255*255 Gassian filter
ksize = 255
sigma = 64
K = 1

x, y = np.mgrid[-(ksize//2):(ksize//2)+1, -(ksize//2):(ksize//2)+1]
gaussian_kernel = K * (np.exp(- ((x**2+y**2) / (2*sigma**2))) )

## Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
plt.imshow(gaussian_kernel, cmap=cm.gray), plt.title("Gaussian kernel")
plt.show()

## Convolve Gassian Kernel with Image
img = cv2.imread("checkerboard1024-shaded.tif") 
blur2 = cv2.filter2D(img, cv2.CV_16S, gaussian_kernel)

result2 = img / blur2

plt.subplot(131),plt.imshow(img)
plt.subplot(132),plt.imshow(blur2)
plt.subplot(133),plt.imshow(result2)
plt.show()
plt.tight_layout()


#================================================================================
# 手刻有沒有跟 opencv 的function結果一樣
print((result == result2).all() == True)  #False
cv2.imshow("result - result2", result - result2)
cv2.waitKey()


"""
test
#%%
import numpy as np
a = np.array([[1, 2, 3], 
              [4, 5, 6]])

b = np.array([[1, 2, 3], 
              [4, 5, 6]])

c = np.array([[1, 2, 3], 
              [4, 5, 7]])

print((a == b).all() == True)   # True
print((a == c).all() == True)   # False
"""
