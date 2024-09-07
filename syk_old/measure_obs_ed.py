import os, sys, argparse, csv
from dynamite import config
from dynamite.states import State
from dynamite.operators import op_sum, op_product, identity, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from dynamite.computations import entanglement_entropy
from itertools import combinations
import numpy as np
from scipy.special import comb

from timeit import default_timer
from datetime import timedelta

msc_dir = '/n/home01/ytan/scratch/deviation_ee/msc_syk/'
seeds = list(range(10))  # Hamiltonian seeds
L = int(sys.argv[1])
TOL = sys.argv[2]

def load_eigvec(L, seed):
    if os.path.exists(os.path.join(msc_dir,f'eigvec_ed_L={L}_seed={seed}_tol={TOL}.metadata')) and \
        os.path.exists(os.path.join(msc_dir,f'eigvec_ed_L={L}_seed={seed}_tol={TOL}.vec')):
        v = State.from_file(os.path.join(msc_dir,f'eigvec_ed_L={L}_seed={seed}_tol={TOL}'))    
        return v
    else:
        print(f'file not found: eigvec_ed_L={L}_seed={seed}_tol={TOL}', flush=True)
        return None
    

def calc_entropy(v, subregion):
    ee = entanglement_entropy(v, subregion)
    return ee

def save_data_long(L, seed, ee, ee_page):
    with open(f'ent_entropy_ed_long.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([L,seed,ee,ee_page])

def save_data(L, ee_avg, ee_page):
    with open(f'ent_entropy_ed.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([L,ee_avg,ee_page])

if __name__ == '__main__':
    ee_avg = 0.
    ee_page = L//2*np.log(2) - 0.5
    for seed in seeds:
        eigvec = load_eigvec(L, seed)
        eigvec.normalize()
        ee = calc_entropy(eigvec, range(L//2))
        ee_avg += ee
        save_data_long(L, seed, ee, ee_page) 
        print(f'seed: [{seed}/{len(seeds)}], ee: {ee}, ee_page: {ee_page}')
    ee_avg /= len(seeds)
    save_data(L, ee_avg, ee_page)