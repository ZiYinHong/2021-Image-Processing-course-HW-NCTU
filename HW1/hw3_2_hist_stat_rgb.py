import cv2
import numpy as np
from matplotlib import pyplot as plt
from itertools import product


img = cv2.imread("hidden_object.jpg")  # shape: (665, 652, 3)
print(img[:,:,0].all()==img[:,:,1].all()==img[:,:,2].all())  #True R=G=B
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

global_mean = np.mean(gray)
global_std = np.std(gray)


# use Histogram Statistics method
kernel_size = 5
output_img = np.zeros_like(img)
#padded_img = cv2.copyMakeBorder(gray, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2,cv2.BORDER_DEFAULT)
rows, cols = img.shape[:-1]
k0, k1, k2, k3 = 0, 0.5, 0, 0.1

for row in np.arange(kernel_size//2+1, rows - (kernel_size//2+1)):  #652
    for col in np.arange(kernel_size//2+1, cols - kernel_size//2+1):
        area = img[(row-(kernel_size//2)) : (row+(kernel_size//2+1)), (col-(kernel_size//2)) : (col+(kernel_size//2+1)), :]
        local_mean = np.mean(area)
        local_std =np.std(area)

        if (local_mean>= k0 and local_mean <= k1 *global_mean) and (local_std >= k2 and local_std <= k3*global_std):
            print("yes")
            output_img[row, col, :] = img[row, col, :]* 20
            #output_img[(row-kernel_size//2) : (row+kernel_size//2), (col-kernel_size//2) : (col+kernel_size//2)] = area*5
        else:
            output_img[row, col, :] = img[row, col, :]
            #output_img[(row-kernel_size//2) : (row+kernel_size//2), (col-kernel_size//2) : (col+kernel_size//2)] = area


# actually similar result as hw3_2.py
print(f"output_img.shape = {output_img.shape}")
cv2.imshow("Histogram Statistics", output_img)
#cv2.imwrite("Histogram_Statistics_1015_rgb.jpg", output_img)

cv2.waitKey()
cv2.destroyAllWindows()
