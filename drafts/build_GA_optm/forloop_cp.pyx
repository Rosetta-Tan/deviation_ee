import numpy as np

def compute():
    A = np.eye(2**6, dtype=np.intc)
    B = np.eye(2**6, dtype=np.intc)
    C = np.eye(2**6, dtype=np.intc)
    D = np.eye(2**6, dtype=np.intc)
    result = np.zeros((2**6, 2**6), dtype=np.intc)
    cdef int i
    for i in range(100000):
        result += A @ B @ C @ D
    return result
    

