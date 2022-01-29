import numpy as np
import matplotlib.pyplot as plt

w = np.exp(2*np.pi/8*-1j)
a = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
              [1, w, -1j, -1j*w, -1, -w, 1j, 1j*w],
              [1, -1j, -1, 1j, 1, -1j, -1, 1j],
              [1, -1j*w, 1j, w, -1, 1j*w, -1j, -w],
              [1, -1, 1, -1, 1, -1, 1, -1],
              [1, -w, -1j, 1j*w, -1, w, 1j, -1j*w],
              [1, 1j, -1, -1j, 1, 1j, -1, -1j],
              [1, 1j*w, 1j, -w, -1, -1j*w, -1j, w]], dtype=complex)

print(f"a.shape : {a.shape}")
print(f"np.real(a).shape : {np.real(a).shape}")
print(f"np.imag(a).shape : {np.imag(a).shape}")

plt.imshow(np.real(a), cmap="gray"), plt.title("np.real(a)")
plt.show()
plt.imshow(np.imag(a), cmap="gray"), plt.title("np.imag(a)")
plt.show()
