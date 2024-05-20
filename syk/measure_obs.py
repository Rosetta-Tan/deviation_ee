import os, sys, csv, argparse
from dynamite import config
from dynamite.states import State
from dynamite.operators import op_sum, op_product, identity, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from dynamite.computations import entanglement_entropy
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
parser.add_argument('--seeds', action=RangeAction, help='range of seeds')
parser.add_argument('--tol', type=float, help='tolerance')
parser.add_argument('--log_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/output/20240425_powermethod_z2_newtol', help='directory to save log files')
parser.add_argument('--vec_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/vec_syk_pm_z2_newtol', help='directory to save state vectors')
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk', help='directory to save msc files')
parser.add_argument('--obs_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/obs_syk', help='directory to save obs files')
parser.add_argument('--gpu', required=False, action='store_true', help='use GPU')
parser.add_argument('--save', type=bool, default=False, help='save data')
args = parser.parse_args()
OBS_CONFIG = {"ent_entropy": {"csv_filepath": os.path.join(args.obs_dir, f'ent_entropy_L={args.L}_tol={args.tol}.csv')},
              "expt_H": {"csv_filepath": os.path.join(args.obs_dir, f'expt_H_L={args.L}_tol={args.tol}.csv')},
              "ent_entropy_f=0.33": {"csv_filepath": os.path.join(args.obs_dir, f'ent_entropy_f=0.33_L={args.L}_tol={args.tol}.csv')}
             }

if args.gpu:
    config.initialize(gpu=True)
config.shell=True  # not using shift-and-invert, so can use shell matrix
config.subspace = Parity('even')
config.L = args.L

def get_hamiltonian(seed):
    filename_str = f'H_L={args.L}_seed={seed}'
    try:
        H = Operator.load(os.path.join(args.msc_dir,f'{filename_str}.msc'))
        logging.info(f'load Hamiltonian: L={args.L}, seed [{seed}/{list(args.seeds)[-1]}] ...')
        return H
    except:
        logging.error(f'Hamiltonian file not found: L={args.L}, seed [{seed}/{list(args.seeds)[-1]}] ...')
        return None

def get_state_vec(seed):
    filename_str = f'v_L={args.L}_seed={seed}_tol={args.tol}'
    try:
        v = State.from_file(os.path.join(args.vec_dir,filename_str))
        print(v.subspace)
        logging.info(f'load state vec: L={args.L}, seed [{seed}/{list(args.seeds)[-1]}] ...')
        return v
    except:
        logging.error(f'state vec file not found: L={args.L}, seed [{seed}/{list(args.seeds)[-1]}] ...')
        return None

def save_data(csv_filepath, seed, data, append=True):
    with open(csv_filepath, 'a' if append else 'w') as file:
        writer = csv.writer(file)
        writer.writerow([args.L, seed, data])

def measure(seed):    
    v = get_state_vec(seed)
    H = get_hamiltonian(seed)
    if v is None or H is None:
        logging.error(f'measurement skipped: L={args.L}, seed [{seed}/{list(args.seeds)[-1]}] ...')
        return None
    
    if "ent_entropy" in OBS_CONFIG.keys():
        if not os.path.exists(OBS_CONFIG["ent_entropy"]["csv_filepath"]):
            with open(OBS_CONFIG["ent_entropy"]["csv_filepath"], 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','ee'])

        ee = entanglement_entropy(v, range(args.L//2))
        logging.info(f'measure ent entropy: L={args.L}, seed [{seed}/{list(args.seeds)[-1]}] ...')
        if args.save:
            csv_filepath = OBS_CONFIG["ent_entropy"]["csv_filepath"]
            save_data(csv_filepath, seed, ee)
    
    if "expt_H" in OBS_CONFIG.keys():
        if not os.path.exists(OBS_CONFIG["expt_H"]["csv_filepath"]):
            with open(OBS_CONFIG["expt_H"]["csv_filepath"], 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','expt_H'])
        
        print("H dim", H.dim)
        expt_H = v.dot(H.dot(v)).real
        logging.info(f'measure expt H: L={args.L}, seed [{seed}/{list(args.seeds)[-1]}] ...')
        if args.save:
            csv_filepath = OBS_CONFIG["expt_H"]["csv_filepath"]
            save_data(csv_filepath, seed, expt_H, append=True)\
    
    if "ent_entropy_f=0.33" in OBS_CONFIG.keys():
        if not os.path.exists(OBS_CONFIG["ent_entropy_f=0.33"]["csv_filepath"]):
            with open(OBS_CONFIG["ent_entropy_f=0.33"]["csv_filepath"], 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','ee'])
        
        ee = entanglement_entropy(v, range(args.L//3))
        logging.info(f'measure ent entropy f=0.33: L={args.L}, seed [{seed}/{list(args.seeds)[-1]}] ...')
        if args.save:
            csv_filepath = OBS_CONFIG["ent_entropy_f=0.33"]["csv_filepath"]
            save_data(csv_filepath, seed, ee)

if __name__ == '__main__':
    for seed in args.seeds:
        measure(seed)
    
