import cv2
from matplotlib import colors, pyplot as plt
import numpy as np

img = cv2.imread("aerial_view.tif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized_img = cv2.equalizeHist(gray)
rows, cols = equalized_img.shape

hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
pdf = hist/equalized_img.size
cdf_255 = np.uint8(np.cumsum(pdf)*255)

cdf_Zq_255 = np.uint8(0.0006*np.cumsum(np.arange(256)**0.4)*255)  # c =  0.0005968222208221872
cdf_Zq_255[-1]  = 255    # 這邊須注意的是，因最後一個累積值算出來是256.35，超過 uint8 範圍因此將它指定為255


# plot result
plt.subplot(221), plt.plot(hist, 'b'), plt.title("equalized_img hist")
plt.subplot(222), plt.plot(pdf, 'b'), plt.legend(), plt.title("equalized_img pdf")
plt.subplot(223), plt.plot(cdf_255, 'b'), plt.title("equalized_img cdf*255")
plt.subplot(224), plt.plot(cdf_Zq_255, 'b'), plt.title("target cdf*255")

plt.tight_layout()
plt.show()


# calculate c parameter
cdf_Zq = np.cumsum(np.arange(256)**0.4)
#print(cdf_Zq_cv2_2[-1])  #1675
print("c = ", 1/cdf_Zq[-1])  # c =  0.0005968222208221872


# implement histogram mapping
output_img = np.zeros_like(equalized_img)
all_vals_in_equalized_img = list(set(equalized_img.ravel()))

for val in all_vals_in_equalized_img:
    if val in cdf_Zq_255:
        index = 0
        for t in (cdf_Zq_255 == val):
            if t == True:
                break
            else:
                index += 1
        pixel_val = index
        indices = (equalized_img == val)
        output_img[indices] = pixel_val
    else:   
        value_abs_diff = [np.abs(int(val) - int(cdf_Zq)) for cdf_Zq in cdf_Zq_255]
        pixel_val = np.argmin(value_abs_diff)           # Will print out smallest index meet the condition
        indices = (equalized_img == val)
        output_img[indices] = pixel_val

# cv2.imshow("output_img_cv2", output_img_cv2)
# # cv2.imwrite("output_img_cv2.jpg", output_img_cv2)
# cv2.waitKey()

#plot result in matplotlib
fig, axs =  plt.subplots(nrows=1, ncols=3)
plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis("off")
plt.subplot(132), plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)), plt.axis("off")
plt.subplot(133), plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)), plt.axis("off")

axs[0].set_title("original img", fontsize=15)
axs[1].set_title("global histogram equlization", fontsize=15)
axs[2].set_title("histogram matching(cv2)", fontsize=15)

plt.tight_layout()
plt.show()


#plot result in matplotlib
fig, axs =  plt.subplots(nrows=1, ncols=2)
output_hist = cv2.calcHist([output_img], [0], None, [256], [0, 256])
output_pdf = output_hist/output_img.size
output_cdf_255 = np.uint8(np.cumsum(output_pdf)*255)

plt.subplot(121), plt.plot(output_pdf) ,plt.title("pdf after histogram matching", fontsize=15)
plt.subplot(122), plt.plot(output_cdf_255), plt.legend(), plt.title("cdf after histogram matching", fontsize=15)
plt.tight_layout()
plt.show()

