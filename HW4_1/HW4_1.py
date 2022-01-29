from scipy.fft import fft
import numpy as np

x = np.array([1, 2, 4, 4])
x = np.array([4, -2, 1, -5, 4, -2, 1, -5])
print("scipy", fft(x))  
print("numpy", np.fft.fft(x))  