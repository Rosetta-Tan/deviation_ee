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
parser.add_argument('--vec_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/vec_syk/powermethod', help='directory to save state vectors')
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk', help='directory to save msc files')
parser.add_argument('--save_concise', type=bool, default=False, help='save concise data')
parser.add_argument('--save_long', type=bool, default=False, help='save long data')
args = parser.parse_args()

def get_avg_entropy(L, save_concise=True, save_long=False):
    ees = []
    for seed in args.seeds:
        filename_str = f'v_L={L}_seed={seed}_tol={args.tol}'
        print(f'collecting state vectors for L={L}, seed [{seed}/{len(args.seeds)}] ...', flush=True)
        if os.path.exists(os.path.join(args.vec_dir,f'{filename_str}.metadata')) and \
            os.path.exists(os.path.join(args.vec_dir,f'{filename_str}.vec')):
            v = State.from_file(os.path.join(args.vec_dir,filename_str))
            ee = calc_entropy(v, range(L//2))
            ees.append(ee)
            if save_long:
                save_data_long(L, seed, ee)
            print(f'seed: [{seed}/{len(args.seeds)}], ee: {ee}')
        else:
            print(f'file not found: {filename_str}', flush=True)
    ee_avg = np.average(ees)
    if save_concise:
        save_data(L, ee_avg)
    return ee_avg

def get_vec(L, seed):
    filename_str = f'v_L={L}_seed={seed}_tol={args.tol}'
    if os.path.exists(os.path.join(args.vec_dir,f'{filename_str}.metadata')) and \
        os.path.exists(os.path.join(args.vec_dir,f'{filename_str}.vec')):
        v = State.from_file(os.path.join(args.vec_dir,filename_str))
        return v
    else:
        print(f'file not found: {filename_str}', flush=True)
        return None

def get_hamiltonian(L, seed):
    filename_str = f'H_L={L}_seed={seed}'
    if os.path.exists(os.path.join(args.msc_dir,f'{filename_str}.msc')):
        H = Operator.load(os.path.join(args.msc_dir,f'{filename_str}.msc'))
        return H
    else:
        print(f'file not found: {filename_str}', flush=True)
        return None
        
def calc_entropy(v, subregion):
    ee = entanglement_entropy(v, subregion)
    return ee

def save_data_long(L, seed, ee, append=True):
    with open(f'ent_entropy_long_tol={args.tol}.csv', 'a' if append else 'w') as file:
        writer = csv.writer(file)
        writer.writerow([L,seed,ee])

def save_data(L, ee_avg, append=True):
    with open(f'ent_entropy_tol={args.tol}.csv', 'a' if append else 'w') as file:
        writer = csv.writer(file)
        writer.writerow([L,ee_avg])

def save_expt_H(L, seed, append=True):
    v = get_vec(args.L, seed)
    H = get_hamiltonian(args.L, seed)
    expt_H = v.dot(H.dot(v)).real
    with open(f'expt_H_abs_tol={args.tol}.csv', 'a' if append else 'w') as file:
        writer = csv.writer(file)
        writer.writerow([L,seed,expt_H])

if __name__ == '__main__':
    # ee_avg = get_avg_entropy(args.L, save_concise=args.save_concise, save_long=args.save_long)
    # print(f'ee_avg: {ee_avg}')
    for seed in args.seeds:
        save_expt_H(args.L, seed, append=True)



