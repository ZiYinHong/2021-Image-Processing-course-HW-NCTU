import cv2
from matplotlib import pyplot as plt

img = cv2.imread("aerial_view.tif")
print(img.shape)
cv2.imshow("aerial_view", img)
cv2.waitKey()
hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

fig, axs =  plt.subplots(nrows=1, ncols=3)

axs[0].plot(hist_b, 'b')
axs[1].plot(hist_g, 'g')
axs[2].plot(hist_r, 'r')

axs[0].set_ylabel('histogram')
axs[0].set_title("blue")
axs[1].set_title("green")
axs[2].set_title("red")

plt.tight_layout()
plt.show()
