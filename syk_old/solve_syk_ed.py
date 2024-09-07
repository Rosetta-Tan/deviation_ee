import os, argparse, csv
from dynamite import config
from dynamite.states import State
from dynamite.subspaces import Parity
from dynamite.operators import op_sum, op_product, identity, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from dynamite.computations import entanglement_entropy
from itertools import combinations
import numpy as np
from scipy.special import comb
from timeit import default_timer
from datetime import timedelta

config.initialize(gpu=True)  # for GPU
# shift and invert is not supported for shell matrices, so can't use config.shell=True

parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, required=True, help='system size')
parser.add_argument('-J', type=float, required=False, default=1.0, help='coupling strength')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--msc_dirc', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk',  help='msc file directory')
parser.add_argument('--out_dirc', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/output/solve_syk_powermethod',  help='output directory')
parser.add_argument('--state_seed_start', type=int, required=False, default=0, help='start seed for random state initialization')
parser.add_argument('--state_seed', type=int, required=False, default=0, help='seed for random state initialization')
parser.add_argument('--tol', type=float, required=False, default=1e-5, help='tolerance for eigensolver')
# parser.add_argument('--state_seed_start', type=int, required=False, default=0, help='start seed for random state initialization')
# parser.add_argument('--state_seed_end', type=int, required=False, default=100, help='end seed for random state initialization')
args = parser.parse_args()
L = args.L
config.L = L
J = args.J
N = 2*L # number of Majoranas
seed = args.seed
msc_dir = args.msc_dirc
if not os.path.isdir(msc_dir):
    os.mkdir(msc_dir)

TOL = args.tol

def load_H(L, seed):
    H = Operator.load(os.path.join(msc_dir,f'H_L={L}_seed={seed}.msc'))
    return H

def solve_H(H, tol=1e-5):
    eigvals, eigvecs = H.eigsolve(getvecs=True, target=0, tol=tol)
    eigval = eigvals[0]
    eigvec = eigvecs[0]
    return eigval, eigvec

if __name__ == '__main__':    
    start = default_timer()
    H = load_H(L, seed)
    H.subspace = Parity('even')
    print(f'load H, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}', flush=True)
    eigval, eigvec = solve_H(H)
    np.savez(os.path.join(msc_dir,f'eigval_ed_L={L}_seed={seed}_tol={TOL}'), eigval=eigval)
    eigvec.save(os.path.join(msc_dir,f'eigvec_ed_L={L}_seed={seed}_tol={TOL}'))
    print(f'solve eigenpairs, L={L}, seed={seed}, tol={TOL}; time elapsed: {timedelta(0,default_timer()-start)}', flush=True)

    
    