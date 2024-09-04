import os
import argparse
import logging
import numpy as np
from dynamite.operators import Operator
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, required=True, help='system size')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_npy_GA', help='output directory')
parser.add_argument('--save', type=bool, required=False, default=True, help='save the output')
args = parser.parse_args()

def load_H(L, seed, msc_dir):
    filename = os.path.join(msc_dir, f'H_L={L}_seed={seed}.msc')
    H = Operator.load(filename)
    H_np = H.to_numpy(sparse=False)
    assert H_np.shape == (2**L, 2**L)
    assert np.allclose(H_np, H_np.T.conj())
    return H_np

def build_G(H, L, seed, save, msc_dir):
    H2 = H @ H
    G = H2 - 1/(2**L) * np.trace(H2) * np.eye(2**L)
    assert np.allclose(np.trace(G), 0), f'trace of G is {np.trace(G)}'
    if save:
        filename = os.path.join(msc_dir, f'G_L={L}_seed={seed}.npy')
        np.save(filename, G)
    return G    

if __name__ == "__main__":
    H = load_H(args.L, args.seed, args.msc_dir)
    G = build_G(H, args.L, args.seed, args.save, args.msc_dir)
    print(f'G_L={args.L}_seed={args.seed} is built and saved.')
    