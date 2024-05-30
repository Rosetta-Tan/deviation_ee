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
parser.add_argument('--LA', type=int, required=True, help='subsystem size')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_GA',  help='output directory')
parser.add_argument('--gpu', type=int, required=False, default=0, help='use GPU')
args = parser.parse_args()

if args.gpu == 1:
    config.initialize(gpu=True)  # for GPU
config.shell=True

N = 2*args.L # total number of Majoranas
NA = 2*args.LA # number of Majoranas in the subsystem
if not os.path.isdir(args.msc_dir):
    os.mkdir(args.msc_dir)

# cache Majoranas to save time
M = [sqrt(2)*majorana(idx) for idx in range(0, NA)]  # TODO: drop the factor of sqrt(2) when the Majorana normalization is fixed
RNG = np.random.default_rng(args.seed)
CPLS = RNG.normal(size=comb(N,4,exact=True))  # the overall constant factor is 1/sqrt((N choose 4))
CPLS_MAP = {(i,j,k,l): CPLS[ind] for ind, (i,j,k,l) in enumerate(combinations(range(N), 4))}  # map from tuple to CPLS index}

def build_GA(save=False):
    ops = []
    coeffs = []
    A_maj_inds = list(range(2*args.LA))  # this is necessary because the iterator should go over Majorana indices instead of spin indices
    normalization_factor = 1.0 / comb(N, 4)
    iterator1 = list(combinations(A_maj_inds, 4))
    iterator2 = list(combinations(A_maj_inds, 4))
    for inds1 in iterator1:  # inds1 is a tuple of 4 indices
        for inds2 in iterator2:
            cdnl = len(set(inds1).union(set(inds2)))
            if cdnl == 6 or cdnl == 8:
                C_cdnl = comb(N, NA) / comb(N-cdnl, NA-cdnl)
                coeffs.append(C_cdnl * CPLS_MAP[inds1] * CPLS_MAP[inds2])
                ops.append(op_product([M[i] for i in inds1] + [M[j] for j in inds2]))
    
    GA = op_sum([c*op for c, op in zip(coeffs, ops)])
    if save == True:
        GA.save(os.path.join(args.msc_dir,f'GA_L={args.L}_LA={args.LA}_seed={args.seed}.msc'))
    return GA

def main():
    logging.info(f'start building GA, L={args.L}, LA={args.LA} seed={args.seed}')
    start = default_timer()
    build_GA(save=True)
    logging.info(f'finish building GA, L={args.L}, LA={args.LA}, seed={args.seed}; time elapsed: {timedelta(0,default_timer()-start)}')
    print(f'finish building GA, L={args.L}, LA={args.LA}, seed={args.seed}; time elapsed: {timedelta(0,default_timer()-start)}')
    
if __name__ == '__main__':
    main()
