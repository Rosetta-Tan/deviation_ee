import cupy as cp
from numba import cuda

# Define the CUDA kernel for matrix multiplication
@cuda.jit
def compute_kernel(result, A, B):
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        temp = 0
        for k in range(A.shape[1]):
            temp += A[i, k] * B[k, j]
        result[i, j] = temp

def matrix_multiply(A, B):
    result = cp.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
    threadsperblock = (16, 16)
    blockspergrid_x = (result.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (result.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    compute_kernel[blockspergrid, threadsperblock](result, A, B)
    cuda.synchronize()
    return result

def compute():
    A = cp.eye(2**6, dtype=int)
    B = cp.eye(2**6, dtype=int)
    C = cp.eye(2**6, dtype=int)
    D = cp.eye(2**6, dtype=int)
    result = cp.zeros((2**6, 2**6), dtype=int)
    
    for _ in range(100000):
        result = matrix_multiply(A, B)
        result = matrix_multiply(result, C)
        result = matrix_multiply(result, D)
    
    return result

if __name__ == '__main__':
    result = compute()
    print(result)