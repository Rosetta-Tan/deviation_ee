from sys import argv
import os
from math import sqrt
from dynamite import config
from dynamite.operators import op_sum, op_product, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from itertools import combinations
import numpy as np
from scipy.special import comb
import argparse

from timeit import default_timer
from datetime import timedelta

# for GPU
config.initialize(slepc_args=['-st_type', 'filter', '-eps_interval', '-0.1,0.1'], gpu=True)  # the EPS interval parameter can be played with
config.shell=True

# for CPU
# config.initialize(slepc_args=['-st_type', 'filter', '-eps_interval', '-0.1,0.1'])
# config.shell=True

parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, required=True, help='system size')
parser.add_argument('-J', type=float, required=False, default=1.0, help='coupling strength')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--dirc', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk',  help='output directory')
args = parser.parse_args()
L = args.L
config.L = L
J = args.J
N = 2*L # number of Majoranas
seed = args.seed
msc_dir = args.dirc
if not os.path.isdir(msc_dir):
    os.mkdir(msc_dir)

# cache Majoranas to save time
M = [sqrt(2)*majorana(idx) for idx in range(0,N)]
RNG = np.random.default_rng(seed)
CPLS = J*sqrt(24)/np.sqrt(N*(N-1)*(N-2)*(N-3)) * RNG.normal(size=comb(N,4,exact=True))  # the overall constant factor is 1/sqrt((N choose 4))

def build_H(save=False):
    ops = [op_product([M[i] for i in idx]) for idx in combinations(range(N), 4)]
    H = op_sum([c*op for c,op in zip(CPLS,ops)])
    if save == True:
        H.save(os.path.join(msc_dir,f'H_{L}_{seed}.msc'))
    return H

def load_H():
    H = Operator.load(os.path.join(msc_dir,f'H_{L}_{seed}.msc'))
    return H

def main():
    start = default_timer()
    # H = build_H(save=True)
    # print(f'build Hamiltonian, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}')
    H = load_H()
    print(f'load Hamiltonian, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}', flush=True)
    evals, evecs = H.eigsolve(getvecs=True, tol=1e-5)
    np.savez(os.path.join('/n/home01/ytan/scratch/deviation_ee/output/solve_polyfilter',f'evals_evecs_{L}_{seed}.npz'), evals=evals, evecs=evecs)
    print(f'polyfilter solve H, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}', flush=True)

if __name__ == '__main__':
    main()
