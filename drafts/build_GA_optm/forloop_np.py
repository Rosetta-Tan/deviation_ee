import numpy as np

def compute():
    A = np.eye(2**6, dtype=np.int32)
    B = np.eye(2**6, dtype=np.int32)
    C = np.eye(2**6, dtype=np.int32)
    D = np.eye(2**6, dtype=np.int32)
    result = np.zeros((2**6, 2**6), dtype=np.int32)
    for _ in range(100000):
        result += A @ B @ C @ D
    return result
    
if __name__ == "__main__":
    print(compute())

