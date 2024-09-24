import os
import argparse
import numpy as np
from itertools import combinations
from scipy.special import comb
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('--L', dest='L', type=int, required=True)
parser.add_argument('--seed', dest='seed', type=int, required=True)
parser.add_argument('--msc_dir', dest='msc_dir', type=str, required=False,
                    default='../data/msc_syk_benchmark')
args = parser.parse_args()

def gen_cpls(L, seed):
    rng = np.random.default_rng(seed)
    cpls = rng.normal(size=comb(2*L, 4, exact=True))
    return cpls

def majorana(idx, L):
    sigma0 = np.eye(2)
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j],[1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    spin_idx = idx // 2
    parity = idx % 2
    if spin_idx == 0:
        if parity == 0:
            m = sigmax
        else:
            m = sigmay
    else:
        m = sigmaz
    # print('first kind', m.shape)
    
    for _ in range(1, spin_idx):
        m = np.kron(sigmaz, m)
        # print('second kind', m.shape)
    
    if spin_idx != 0:
        if parity == 0:
            m = np.kron(sigmax, m)
        else:
            m = np.kron(sigmay, m)
    # print('third kind', m.shape)
    
    for _ in range(spin_idx+1, L):
        m = np.kron(sigma0, m)
    
    # print('fourth kind', m.shape)
    # print(f'idx={idx}, L={L}, m.shape={m.shape}')

    return m

def build_syk(L, seed):
    cpls = gen_cpls(L, seed)
    M = [majorana(i, L) for i in range(2*L)]
    inds = list(combinations(range(2*L), 4))
    assert len(cpls) == len(inds), f'len(cpls)={len(cpls)} != len(inds)={len(inds)}'
    H = np.zeros((2**L, 2**L), dtype=np.complex128)
    start = timer()
    for i in range(len(cpls)):
        end = timer()
        print(f'Processing {i}th term, time elapsed: {end-start:.2f}s')
        # print(M[inds[i][0]].shape, M[inds[i][1]].shape, M[inds[i][2]].shape, M[inds[i][3]].shape)
        op = cpls[i] * M[inds[i][0]] @ M[inds[i][1]] @ M[inds[i][2]] @ M[inds[i][3]]
        assert H.shape == op.shape, f'H.shape={H.shape} != op.shape={op.shape}'
        H += op
    H *= 1./np.sqrt(comb(2*L, 4))
    end = timer()
    print(f'Total time elapsed: {end-start:.2f}s')
    return H

if __name__ == "__main__":
    L = args.L
    seed = args.seed
    args.msc_dir = 'data'
    H = build_syk(L, seed)
    np.save(os.path.join(args.msc_dir, f'H_L={L}_seed={seed}.npy'), H)