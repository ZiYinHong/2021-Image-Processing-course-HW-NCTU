import cv2
import numpy as np
from matplotlib import pyplot as plt
from itertools import product


img = cv2.imread("hidden_object_2.jpg")  # shape: (665, 652, 3)
# print(img[:,:,0].all()==img[:,:,1].all()==img[:,:,2].all())  #True R=G=B
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized_img = cv2.equalizeHist(gray)



cv2.imshow("img", img)
cv2.imshow("global_histogram_equalization", equalized_img)
#cv2.imwrite("global_histogram_equalization.jpg", equalized_img)


# use local enhancement method
kernel_size = 5
output_img = np.zeros_like(gray)
#padded_img = cv2.copyMakeBorder(gray, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2,cv2.BORDER_DEFAULT)
rows, cols = gray.shape

for row in np.arange(kernel_size//2+1, rows - (kernel_size//2+1)):  #652
    for col in np.arange(kernel_size//2+1, cols - kernel_size//2+1):
        area = gray[(row-(kernel_size//2)) : (row+(kernel_size//2+1)), (col-(kernel_size//2)) : (col+(kernel_size//2+1))]
        if area.size == kernel_size*kernel_size:
            equalized_area = cv2.equalizeHist(area)
            output_img[row, col] = equalized_area[kernel_size//2, kernel_size//2]   # (kernel_size, kernel_size) 中心的那個像素值
        else:
            output_img[row, col] = gray[row, col]

cv2.imshow("local_histogram_equalization", output_img)
#cv2.imwrite("local_histogram_equalization.jpg", output_img)
cv2.waitKey()
cv2.destroyAllWindows()

