import numpy as np
def DFT2d(f): 
    """
     f is a two dimansional, one channel image
    """
    M, N  = f.shape
    F = np.zeros((M,N),dtype=complex)   # so important!!! datatype must be in complex

    ## return DFT image F(u, v)   
    for u in range(0, M):
        for v in range(0, N):
            print(f"u = {u}, v = {v}")
            for x in range(0, M):
                for y in range(0, N):
                    F[u, v] += f[x, y]*np.exp(-2j* np.pi* (u*x/M + v*y/N))
    return F