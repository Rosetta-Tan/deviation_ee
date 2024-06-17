import os, logging
from math import sqrt
from dynamite import config
from dynamite.operators import op_sum, op_product, Operator, zero
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
parser.add_argument('--n_groups', type=int, required=True, \
                    help='number of groups to divide the outerloop')
parser.add_argument('--group_idx', type=int, required=True, \
                    help='index of the group to build')
parser.add_argument('--msc_dir', type=str, required=False, \
                    default='/n/home01/ytan/scratch/deviation_ee/msc_npy_GA', \
                        help='output directory')
parser.add_argument('--gpu', type=int, required=False, default=0, \
                    help='use GPU')
parser.add_argument('--save', type=bool, required=False, default=False, \
                    help='save the result')
parser.add_argument('--save_dir', type=str, required=False, \
                    default='/n/home01/ytan/scratch/deviation_ee/msc_npy_GA', \
                        help='output directory')
args = parser.parse_args()

if args.gpu == 1:
    config.initialize(gpu=True)
# config.shell=True

N = 2*args.L # total number of Majoranas
NA = 2*args.LA # number of Majoranas in the subsystem
if not os.path.isdir(args.msc_dir):
    os.mkdir(args.msc_dir)

# cache Majoranas to save time
M = [majorana(idx) for idx in range(0, NA)]  # TODO: drop the factor of sqrt(2) when the Majorana normalization is fixed
RNG = np.random.default_rng(args.seed)
CPLS = RNG.normal(size=comb(N, 4,exact=True))  # the overall constant factor is 1/sqrt((N choose 4))

def build_GA_group(group_idx, n_groups):
    logging.info(f'start building GA, L={args.L}, LA={args.LA} seed={args.seed}')
    start = default_timer()
    
    op_group = zero()
    ops = []
    coeffs = []
    A_maj_inds = list(range(2*args.LA))  # this is necessary because the iterator should go over Majorana indices instead of spin indices
    iterator = list(combinations(A_maj_inds, 4))
    
    # divide the outerloop into n_groups parts
    if group_idx == n_groups-1:
        iterator_group = iterator[group_idx*len(iterator)//n_groups:]
    else:
        iterator_group = iterator[group_idx*len(iterator)//n_groups: \
            (group_idx+1)*len(iterator)//n_groups]
    # inds1 is a tuple of 4 indices
    for i1, inds1 in enumerate(iterator_group, \
        start=group_idx*len(iterator)//n_groups):
        for i2, inds2 in enumerate(iterator):
            cdnl = len(set(inds1).union(set(inds2)))
            if cdnl == 6 or cdnl == 8:
                C_cdnl = comb(N, NA) / comb(N-cdnl, NA-cdnl)
                coeffs.append(C_cdnl * CPLS[i1] * CPLS[i2])
                ops.append(op_product([M[i] for i in inds1] + \
                    [M[j] for j in inds2]))

    op_group = op_sum([c*op for c, op in zip(coeffs, ops)])
    op_group = op_group.to_numpy(sparse=False)

    logging.info(f'finish building GA, L={args.L}, \
                 LA={args.LA}, seed={args.seed}; \
                 time elapsed: {timedelta(0,default_timer()-start)}')
    return op_group

if __name__ == '__main__':
    n_groups = args.n_groups
    pipeline(args.group_idx, args.n_groups)