from sys import argv
import os, logging
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
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/old/msc_syk',  help='output directory')
parser.add_argument('--gpu', type=int, required=False, default=0, help='use GPU')
args = parser.parse_args()

if args.gpu == 1:
    config.initialize(gpu=True)  # for GPU
config.shell=True

L = args.L
config.L = L
J = args.J
N = 2*L # number of Majoranas
seed = args.seed
msc_dir = args.msc_dir
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
        H.save(os.path.join(msc_dir,f'H_L={L}_seed={seed}.msc'))
    return H

def main():
    logging.info(f'start building Hamiltonian, L={L}, seed={seed}')
    start = default_timer()
    H = build_H(save=True)
    logging.info(f'finish building Hamiltonian, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}')
    print(f'finish building Hamiltonian, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}')
    
if __name__ == '__main__':
    main()
