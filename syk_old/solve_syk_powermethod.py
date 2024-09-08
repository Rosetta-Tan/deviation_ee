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
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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
parser.add_argument('--L', type=int, required=True, help='system size')
parser.add_argument('--J', type=float, required=False, default=1.0, help='coupling strength')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/old/msc_syk',  help='msc file directory')
parser.add_argument('--log_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/old/output',  help='log file directory')
parser.add_argument('--eval_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/old/extrm_eigval',  help='eigenvalue file directory')
parser.add_argument('--vec_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/old/vec_syk/powermethod',  help='vector file directory')   # for saving the evolved state
parser.add_argument('--tol', type=float, required=False, default=1e-9, help='tolerance coefficient of energy variance of states for power method')
parser.add_argument('--gpu', required=False, action='store_true', help='use GPU')
args = parser.parse_args()

# for GPU
if args.gpu:
    config.initialize(gpu=True)
config.shell=True  # not using shift-and-invert, so can use shell matrix
config.subspace = Parity('even')

L = args.L
config.L = L
J = args.J
N = 2*L # number of Majoranas
seed = args.seed
tol = args.tol
if not os.path.isdir(args.msc_dir):
    os.mkdir(args.msc_dir)
if not os.path.isdir(args.vec_dir):
    os.mkdir(args.vec_dir)

def get_extrm_eigval(L, seed, eval_dir=args.eval_dir):
    filepath = os.path.join(eval_dir, f'eval_L={L}_seed={seed}.npy')
    # eval is a 1D numpy array
    eval = np.load(filepath)
    return eval[0]

LAMBDA = abs(get_extrm_eigval(L, seed))+0.001  # the tolerance was set to ~1e-5, so we add 0.001 to the largest eigenvalue

def cal_energy_variance(H, v):
    Hv = H.dot(v)
    return Hv.dot(Hv).real

def load_H(L, seed):
    if not os.path.exists(os.path.join(args.msc_dir,f'H_L={L}_seed={seed}.msc')):
        logging.error(f'H_L={L}_seed={seed}.msc does not exist.')
    H = Operator.load(os.path.join(args.msc_dir,f'H_L={L}_seed={seed}.msc'))
    return H

def diff_state_vecs(v0, v1):
    v_diff = v1.copy()
    v_diff.axpy(-1.0, v0)
    return v_diff

def power_method_like_evol(H, op1, op2, v0, tol):
    # initialize
    v0.normalize()
    num_pm_steps = 0
    logging.debug(f'num_pm_steps: {num_pm_steps}, energy variance: {cal_energy_variance(H, v0)}')
    v1 = op1.dot(v0)
    v2 = op2.dot(v1)
    v2.normalize()
    num_pm_steps += 1
    logging.debug(f'num_pm_steps: {num_pm_steps}, energy variance: {cal_energy_variance(H, v2)}')
    while cal_energy_variance(H, v2) > tol:
        v0, v2 = v2, v0  # swap the two states
        v1 = op1.dot(v0)
        v2 = op2.dot(v1)
        v2.normalize()
        num_pm_steps += 1
        if num_pm_steps % 100 == 0:
            logging.debug(f'num_pm_steps: {num_pm_steps}, energy variance: {cal_energy_variance(H, v2)}')
    return v2, num_pm_steps

# def power_method_debug(H, op1, op2, v0, tol):
#     # initialize
#     v0.normalize()
#     v1 = op1.dot(v0)
#     logging.debug(f'dynamite mode debug - state energy variance at step 0.5: {cal_energy_variance(H, v1)}')
#     v2 = op2.dot(v1)
#     v2.normalize()
#     logging.debug(f'dynamite mode debug - state energy expt val at step 0: {v0.dot(H.dot(v0)).real}')
#     logging.debug(f'dynamite mode debug - state energy variance at step 0: {cal_energy_variance(H, v0)}')
#     logging.debug(f'dynamite mode debug - state overlap between step 0 and step 1: {abs(v0.dot(v2))}')
#     logging.debug(f'dynamite mode debug - state energy expt val at step 1: {v2.dot(H.dot(v2)).real}')
#     logging.debug(f'dynamite mode debug - state energy variance at step 1: {cal_energy_variance(H, v2)}')
#     # diff_mat = op1 + H - LAMBDA*identity()
#     # logging.debug(f'dynamite mode debug - the difference matrix inf-norm: {diff_mat.infinity_norm()}')

#     H_np = H.to_numpy()
#     op1_np = op1.to_numpy()
#     np.save(os.path.join(args.vec_dir,f'op1_L={L}_seed={seed}_tol={tol}.npy'), op1_np)
#     diff_mat = op1_np + H_np - LAMBDA*sp.identity(2**config.L)
#     logging.debug(f'numpy mode debug - the difference matrix 2-norm: {spla.norm(diff_mat, ord=2)}')
#     op2_np = op2.to_numpy()
#     v0_np = v0.to_numpy()
#     v1_np = op1_np @ v0_np
#     logging.debug(f'numpy mode debug - state energy variance at step 0.5: {(v1_np.conj().T @ H_np @ H_np @ v1_np).real}')
#     v2_np = op2_np @ v1_np
#     v2_np /= np.linalg.norm(v2_np)
#     logging.debug(f'numpy mode debug - state energy variance at step 0: {(v0_np.conj().T @ H_np @ H_np @ v0_np).real}')
#     logging.debug(f'numpy mode debug - state overlap between step 0 and step 1: {abs(v0_np.conj().T @ v2_np)}')
#     logging.debug(f'numpy mode debug - state energy variance at step 1: {(v2_np.conj().T @ H_np @ H_np @ v2_np).real}')
    
#     return v2, 1

def main():
    start = default_timer()
    H = (1/4) * load_H(L=L, seed=seed)  # the factor 1/4 is because I used the wrong normalization when building the Hamiltonian
    logging.info(f'load Hamiltonian, L={L}, seed={seed}, to be solved at tol {tol}; time elapsed: {timedelta(seconds=default_timer()-start)}')
    op1 = (LAMBDA/4)*identity() - H  # the same 1/4 factor here
    op2 = (LAMBDA/4)*identity() + H  # the same 1/4 factor here
    logging.info(f'construct the two operators involved in power method, L={L}, seed={seed}; time elapsed: {timedelta(seconds=default_timer()-start)}')
    
    # check whether v0 has been saved
    if os.path.exists(os.path.join(args.vec_dir,f'v0_L={L}_seed={seed}_tol={tol}.metadata')) and \
        os.path.exists(os.path.join(args.vec_dir,f'v0_L={L}_seed={seed}_tol={tol}.vec')):
        logging.info(f'v0 exists, L={L}, seed={seed}, tol={tol}; time elapsed: {timedelta(seconds=default_timer()-start)}')
        v0 = State.from_file(os.path.join(args.vec_dir,f'v0_L={L}_seed={seed}_tol={tol}'))
    else:
        v0 = State(state='random', seed=0)
        v0.save(os.path.join(args.vec_dir,f'v0_L={L}_seed={seed}_tol={tol}'))

    # a sanity check
    logging.debug(f'dimension of v0: len(v0)={len(v0)}; expetced dimension: 2^(L-1)={2**(config.L-1)}; matched: {len(v0) == 2**(config.L-1)}')

    v, num_pm_steps = power_method_like_evol(H, op1, op2, v0, tol=tol)
    # v, num_pm_steps = power_method_debug(H, op1, op2, v0, tol)  # for debugging purpose
    logging.info(f'evolve state with power method, L={L}, seed={seed}; pm_step: {num_pm_steps}, time elapsed: {timedelta(seconds=default_timer()-start)}')

    # save data
    v.save(os.path.join(args.vec_dir,f'v_L={L}_seed={seed}_tol={tol}'))
    print(f'L={L}, seed={seed}, tol={tol}, pm_step={num_pm_steps}', flush=True)

if __name__ == '__main__':
    main()
