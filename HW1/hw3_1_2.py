import cv2
from matplotlib import pyplot as plt

img = cv2.imread("aerial_view.tif")

# Note: When performing histogram equalization with OpenCV, 
# we must supply a grayscale/single-channel image.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized_img = cv2.equalizeHist(gray)

# calculate histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

# plot result
fig, axs =  plt.subplots(nrows=2, ncols=2)
plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis("off")
plt.subplot(222), plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)), plt.axis("off")
axs[1, 0].plot(hist)
axs[1, 1].plot(equalized_hist)

axs[0, 0].set_ylabel('graph')
axs[1, 0].set_ylabel('histogram')

axs[0, 0].set_title("before histogram equlization", fontsize=10)
axs[0, 1].set_title("after histogram equlization", fontsize=10)

plt.tight_layout()
plt.show()
