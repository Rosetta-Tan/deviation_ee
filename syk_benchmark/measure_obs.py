import os, sys, csv, argparse
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
import logging

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[FlushStreamHandler()])

# Custom action to convert input to a range
class RangeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Split the input string on "-" to get start and end of range
        start, end = map(int, values.split('-'))
        # Store the range in the specified namespace
        setattr(namespace, self.dest, range(start, end))

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, help='system size')
parser.add_argument('--seed', type=int, help='sample seed')
parser.add_argument('--tol', type=float, help='tolerance')
parser.add_argument('--log_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/output/20240425_powermethod_z2_newtol', help='directory to save log files')
parser.add_argument('--vec_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/vec_syk_pm_z2_newtol', help='directory to save state vectors')
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk', help='directory to save msc files')
parser.add_argument('--obs_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/obs_syk', help='directory to save obs files')
parser.add_argument('--gpu', required=False, action='store_true', help='use GPU')
parser.add_argument('--save', type=bool, default=False, help='save data')
args = parser.parse_args()
args.obs_dir = 'data'
OBS_CONFIG = {"ent_entropy": {"csv_filepath": os.path.join(args.obs_dir, f'ent_entropy_L={args.L}_seed={args.seed}_tol={args.tol}.csv')},
              "expt_H": {"csv_filepath": os.path.join(args.obs_dir, f'expt_H_L={args.L}_seed={args.seed}_tol={args.tol}.csv')},
              "expt_H2": {"csv_filepath": os.path.join(args.obs_dir, f'expt_H2_L={args.L}_seed={args.seed}_tol={args.tol}.csv')},
              "ent_entropy_f=0.33": {"csv_filepath": os.path.join(args.obs_dir, f'ent_entropy_f=0.33_L={args.L}_seed={args.seed}_tol={args.tol}.csv')}
             }

if args.gpu:
    config.initialize(gpu=True)
config.shell=True  # not using shift-and-invert, so can use shell matrix
# config.subspace = Parity('even')
config.L = args.L

def gen_projector(L):
    P = np.zeros((2**(L-1), 2**L))
    for i in range(2**L):
        if bin(i).count('1') % 2 == 0:
            P[i//2, i] = 1    
    return P

def project_op(op, P):
    L = int(np.log2(op.shape[1]))
    op_new = P @ op @ P.T
    assert op_new.shape == (2**(L-1), 2**(L-1))
    return op_new

def extend_state(v):
    L = int(np.log2(v.shape[0])) + 1
    v_ext = np.zeros(2**L, dtype=np.complex128)
    for i in range(2**L):
        if bin(i).count('1') % 2 == 0:
            v_ext[i] = v[i//2]
    return v_ext

def get_state_vec(L, seed, tol):
    filename_str = f'v_L={L}_seed={seed}_tol={tol}'
    try:
        v = State.from_file(os.path.join(args.vec_dir,filename_str))
        print(v.subspace)
        logging.info(f'load state vec: L={args.L}, seed {seed} ...')
        return v
    except:
        logging.error(f'state vec file not found: L={args.L}, seed {seed} ...')
        return None

def get_ee(v, L, keep):
    discard = list(set(range(L)) - set(keep))
    rdm = np.zeros((2**len(keep), 2**len(keep)), dtype=np.complex128)
    for i in range(2**len(keep)):
        for j in range(2**len(keep)):
            for k in range(2**len(discard)):
                idx1 = (i << len(discard)) + k
                idx2 = (j << len(discard)) + k
                rdm[i,j] += v[idx1].conj()*v[idx2]
    _, s, _ = np.linalg.svd(rdm)
    ee = -np.sum(s*np.log(s))
    return ee

def save_data(L, seed, data, csv_filepath, append=True):
    if append == False:
        with open(csv_filepath, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['L','seed','data'])
            writer.writerow([L, seed, data])
    else:
        with open(csv_filepath, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([L, seed, data])

if __name__ == '__main__':
    L = args.L
    seed = args.seed
    tol = args.tol
    args.vec_dir = 'data'
    v = np.load(os.path.join(args.vec_dir, f'v_L={L}_seed={seed}_tol={tol}.npy'))
    assert v.shape[0] == 2**(L-1)
    v = extend_state(v)
    args.msc_dir = 'data'
    H = np.load(os.path.join(args.msc_dir, f'H_L={L}_seed={seed}.npy'))
    assert H.shape == (2**L, 2**L)
    
    if "ent_entropy" in OBS_CONFIG.keys():
        if not os.path.exists(OBS_CONFIG["ent_entropy"]["csv_filepath"]):
            with open(OBS_CONFIG["ent_entropy"]["csv_filepath"], 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','ee'])

        ee = get_ee(v, L, range(L//2))
        logging.info(f'measure ent entropy: L={L}, seed {seed} ...')
        if args.save:
            csv_filepath = OBS_CONFIG["ent_entropy"]["csv_filepath"]
            save_data(L, seed, ee, csv_filepath, append=False)
    
    if "expt_H" in OBS_CONFIG.keys():
        if not os.path.exists(OBS_CONFIG["expt_H"]["csv_filepath"]):
            with open(OBS_CONFIG["expt_H"]["csv_filepath"], 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','expt_H'])
        
        print("H dim", H.shape)
        Hv = H @ v
        expt_H = np.dot(v.conj().T, Hv).real
        logging.info(f'measure expt H: L={L}, seed {seed} ...')
        if args.save:
            csv_filepath = OBS_CONFIG["expt_H"]["csv_filepath"]
            save_data(L, seed, expt_H, csv_filepath, append=False)

    if "expt_H2" in OBS_CONFIG.keys():
        if not os.path.exists(OBS_CONFIG["expt_H2"]["csv_filepath"]):
            with open(OBS_CONFIG["expt_H2"]["csv_filepath"], 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','expt_H2'])
        
        Hv = H @ v
        expt_H2 = np.dot(Hv.conj().T, Hv).real
        logging.info(f'measure expt H2: L={L}, seed {seed} ...')
        if args.save:
            csv_filepath = OBS_CONFIG["expt_H2"]["csv_filepath"]
            save_data(L, seed, expt_H2, csv_filepath, append=False)
    
    if "ent_entropy_f=0.33" in OBS_CONFIG.keys():
        if not os.path.exists(OBS_CONFIG["ent_entropy_f=0.33"]["csv_filepath"]):
            with open(OBS_CONFIG["ent_entropy_f=0.33"]["csv_filepath"], 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','ee'])
        
        ee = get_ee(v, L, range(L//3))
        logging.info(f'measure ent entropy f=0.33: L={L}, seed {seed} ...')
        if args.save:
            csv_filepath = OBS_CONFIG["ent_entropy_f=0.33"]["csv_filepath"]
            save_data(L, seed, ee, csv_filepath, append=False)