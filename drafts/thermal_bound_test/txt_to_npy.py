import numpy as np
import os

def test_GA(GA, LA):
    assert GA.shape == (2**LA, 2**LA)
    assert np.allclose(GA, GA.conj().T)
    print(f'trace(GA^2)/2^LA: {np.trace(GA @ GA)/2**LA}')
    assert np.allclose(np.trace(GA), 0, atol=1e-4), f"trace(GA) = {np.trace(GA)}"

if __name__ == "__main__":
    L = 12
    LA = L//2
    seed = 2

    GA_real_filebase = f"GA_real_L={L}_LA={LA}_seed={seed}"
    GA_imag_filebase = f"GA_imag_L={L}_LA={LA}_seed={seed}"
    GA_filebase = f"GA_fromcpp_L={L}_LA={LA}_seed={seed}"

    with open(f"{GA_real_filebase}.txt") as f:
        GA_real = np.array([list(map(float, line.split())) for line in f])
        
    with open(f"{GA_imag_filebase}.txt") as f:
        GA_imag = np.array([list(map(float, line.split())) for line in f])
        
    GA = GA_real + 1j*GA_imag

    test_GA(GA, LA)

    np.save(f"{GA_filebase}.npy", GA)