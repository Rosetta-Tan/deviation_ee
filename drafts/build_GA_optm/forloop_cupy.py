from numba import cuda
from numba import jit
import numpy as np
import cupy as cp

@jit
def compute():
    A = cp.eye(2**6, dtype=int)
    B = cp.eye(2**6, dtype=int)
    C = cp.eye(2**6, dtype=int)
    D = cp.eye(2**6, dtype=int)
    result = cp.zeros((2**6, 2**6), dtype=int)
    
    for _ in range(100000):
        result += A @ B @ C @ D
    
    return result

if __name__ == '__main__':
    print(compute())