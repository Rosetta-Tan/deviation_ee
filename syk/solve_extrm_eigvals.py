from sys import argv
import os
from math import sqrt
from dynamite import config
from dynamite.operators import op_sum, op_product, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from dynamite.subspaces import Parity
from itertools import combinations
import numpy as np
from scipy.special import comb
import argparse
from timeit import default_timer
from datetime import timedelta
import logging

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[FlushStreamHandler()])

parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, required=True, help='system size')
parser.add_argument('-J', type=float, required=False, default=1.0, help='coupling strength')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk',  help='msc file directory')
parser.add_argument('--log_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/output/solve_extrm_eigvals',  help='log directory')
parser.add_argument('--res_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/output/solve_extrm_eigvals',  help='output directory')
parser.add_argument('--gpu', required=False, action='store_true', help='use GPU')
args = parser.parse_args()

# for GPU
if args.gpu:
    config.initialize(gpu=True)  # the EPS interval parameter can be played with
config.shell=True
config.subspace = Parity('even')

L = args.L
config.L = L
J = args.J
N = 2*L # number of Majoranas
seed = args.seed
if not os.path.isdir(args.msc_dir):
    os.mkdir(args.msc_dir)
if not os.path.isdir(args.res_dir):
    os.mkdir(args.res_dir)

def load_H():
    H = Operator.load(os.path.join(args.msc_dir,f'H_L={L}_seed={seed}.msc'))
    return H

def main():
    start = default_timer() 
    H = load_H()
    logging.info(f'load Hamiltonian, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}')
    evals = H.eigsolve(which='exterior', tol=1e-5)    
    logging.info(f'ED solve H extremal eigval, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}')
    
    # save data
    np.savez(os.path.join(args.log_dir, f'evals_L={L}_seed={seed}.npz'), evals=evals)
    np.savez(os.path.join(args.res_dir, f'evals_L={L}_seed={seed}.npz'), evals=evals)
    print(f'L={L}, seed={seed}, evals={evals}')

if __name__ == '__main__':
    main()
