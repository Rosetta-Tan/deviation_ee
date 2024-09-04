import argparse
import numpy as np
from scipy.special import comb

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, required=True, help='system size')
parser.add_argument('--LA', type=int, required=True, help='subsystem size')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
args = parser.parse_args()

N = 2*args.L
NA = 2*args.LA
RNG = np.random.default_rng(args.seed)
CPLS = list(RNG.normal(size=comb(NA, 4, exact=True)))

filename = f'cpls_L={args.L}_LA={args.LA}_seed={args.seed}.txt'
np.savetxt(filename, CPLS)
