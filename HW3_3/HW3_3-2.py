import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

img = cv2.imread("N1.bmp") 
img = img[:,:,::-1]
print(img)
print(img.max())
#print(img.shape)  #(480, 640, 3)
# cv2.imshow("img", img)
# cv2.waitKey()
plt.imshow(img)
plt.show()

blur = cv2.GaussianBlur(img, (221, 161), sigmaX= 37, sigmaY = 27, borderType = cv2.BORDER_REFLECT)  # /6
#blur = cv2.GaussianBlur(img, (221, 161), sigmaX= 55, sigmaY = 40, borderType = cv2.BORDER_REFLECT)  # /4 similar
blur = blur  + 70 # 不加會太亮
print(blur.max())
plt.imshow(blur)
plt.show()


result = img / (blur)
print(f"result.min(), result.max() = {result.min(), result.max()}")
plt.imshow(result)
plt.show()