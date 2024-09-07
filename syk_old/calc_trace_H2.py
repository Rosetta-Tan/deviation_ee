import os, argparse, csv
from dynamite import config
from dynamite.states import State
from dynamite.operators import op_sum, op_product, identity, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from dynamite.subspaces import Parity
from itertools import combinations
import numpy as np
from scipy.special import comb
from timeit import default_timer
from datetime import timedelta
import sys
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
parser.add_argument('--log_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/log_syk',  help='log file directory')
parser.add_argument('--res_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/tr_H2',  help='result file directory')   # for saving data
parser.add_argument('--gpu', required=False, action='store_true', help='use GPU')
parser.add_argument('--benchmark', type=int, required=False, default=0, help='0 for no benchmarking, 1 for benchmarking')
args = parser.parse_args()

# for GPU
if args.gpu:
    config.initialize(gpu=True)
config.shell=True  # not using shift-and-invert, so can use shell matrix

L = args.L
config.L = L
J = args.J
N = 2*L # number of Majoranas
seed = args.seed
if not os.path.isdir(args.msc_dir):
    os.mkdir(args.msc_dir)
if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
if not os.path.isdir(args.res_dir):
    os.mkdir(args.res_dir)


def load_H(L, seed):
    if not os.path.exists(os.path.join(args.msc_dir,f'H_L={L}_seed={seed}.msc')):
        logging.error(f'H_L={L}_seed={seed}.msc does not exist.')
    H = Operator.load(os.path.join(args.msc_dir,f'H_L={L}_seed={seed}.msc'))
    return H

def calc_tr_h2(h):
    """
    Calculate the trace of h^2, but avoid calculating h^2 explicitly.
    The v_sp used is a special state vector of all ones (normalized).
    """
    v_sp = State(state='uniform')
    v_sp.normalize()
    hv = h.dot(v_sp)
    return 2**config.L * hv.dot(hv)
    
def cal_tr_h2_exact(h):
    """
    Calculate the trace of h^2 using numpy.
    Only call this function at small system size for benchmarking.
    """
    h_np = h.to_numpy(sparse=False)
    return np.trace(h_np @ h_np)

def main():
    start = default_timer()
    H = load_H(L=L, seed=seed)
    logging.info(f'load Hamiltonian, L={L}, seed={seed}; time elapsed: {timedelta(seconds=default_timer()-start)}')
    
    tr_h2 = calc_tr_h2(H)
    print(f'tr(H^2) = {tr_h2}', flush=True)
    logging.info(f'calculated tr(H^2); time elapsed: {timedelta(seconds=default_timer()-start)}')

    if args.benchmark:
        tr_h2_exact = cal_tr_h2_exact(H)
        print(f'tr(H^2) (exact) = {tr_h2_exact}', flush=True)
        print(f'error = {np.abs(tr_h2-tr_h2_exact)}', flush=True)
        logging.info(f'benchmarking complete; time elapsed: {timedelta(seconds=default_timer()-start)}')

    # save results
    with open(os.path.join(args.res_dir, f'tr_H2_L={L}_seed={seed}.txt'), 'w') as file:
        file.write(f'{tr_h2}')

if __name__ == '__main__':
    main()




