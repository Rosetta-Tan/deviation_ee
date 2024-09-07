import cupy as cp
import numpy as np
import argparse
from itertools import combinations
from scipy.special import comb
import logging
import psutil
import os


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[FlushStreamHandler()])

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, help='system size')
parser.add_argument('--LA', type=int, help='subsystem size')
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--save', type=bool, required=False, default=False, \
                    help='save GA')
parser.add_argument('--save_dir', type=str, required=False, default='./', \
                    help='save directory')
args = parser.parse_args()


def gen_maj(L) -> cp.ndarray:
    sigmax = cp.array([[0, 1.], [1., 0]], dtype=complex)
    sigmay = cp.array([[0, -1.j], [1.j, 0]], dtype=complex)
    sigmaz = cp.array([[1., 0], [0, -1.]], dtype=complex)
    sigma0 = cp.array([[1., 0], [0, 1.]], dtype=complex)
    majorana_list = []

    for i_site in range(L):
        if i_site==0:
            maj_even = sigmax
            maj_odd = sigmay
        else:
            maj_even = sigmaz
            maj_odd = sigmaz

        for j_site in range(1, L): 
            if j_site < i_site:
                maj_even = cp.kron(maj_even, sigmaz)
                maj_odd = cp.kron(maj_odd, sigmaz)
            elif j_site == i_site: 
                maj_even = cp.kron(maj_even, sigmax)
                maj_odd = cp.kron(maj_odd, sigmay)
            else:
                maj_even = cp.kron(maj_even, sigma0)
                maj_odd = cp.kron(maj_odd, sigma0)

        majorana_list += [maj_even, maj_odd]

    return majorana_list

def op_product(ops) -> cp.ndarray:
    """
    Compute the product of operators.
    """
    op = ops[0]
    for i in range(1, 8):
        op = op.dot(ops[i])
    return op  # convert to cupy.ndarray

def build_GA(majorana, L: int, LA: int, cpls: np.ndarray,
    iterator: np.ndarray) -> cp.ndarray:
    """
    Params:
        majorana: list of majorana operators
        cpls: coupling constants
        iterator: all allowed combinations of Majorana indices for single copy
    Return:
        GA = sum_{J_1, J_2} C_{J_1\cup J_2} M_{J_1} M_{J_2}
    """
    N = 2 * L
    NA = 2 * LA
    normalization_factor = 4*3*2*1 / N*(N-1)*(N-2)*(N-3)

    GA = cp.zeros((2**LA, 2**LA), dtype=complex)
    for i1, inds1 in enumerate(iterator):
        for i2, inds2 in enumerate(iterator):
            cdnl = len(np.unique(
                np.concatenate((inds1, inds2))))  # cardinality
            if cdnl == 6:
                C_cdnl = N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5) / \
                    NA*(NA-1)*(NA-2)*(NA-3)*(NA-4)*(NA-5)
                coeff = C_cdnl * cpls[i1] * cpls[i2]

                GA += coeff * op_product([majorana[i] for i in inds1] + \
                    [majorana[j] for j in inds2])
            elif cdnl == 8:
                C_cdnl = N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5)*(N-6)*(N-7) / \
                    NA*(NA-1)*(NA-2)*(NA-3)*(NA-4)*(NA-5)*(NA-6)*(NA-7)
                coeff = C_cdnl * cpls[i1] * cpls[i2]
                GA += coeff * op_product([majorana[i] for i in inds1] + \
                    [majorana[j] for j in inds2])
    
    return GA * normalization_factor

def pipeline():
    # Set up memory monitoring
    process = psutil.Process()
    logging.info(f"Initial memory usage: \
        {process.memory_info().rss / 1024 / 1024} MB")

    logging.info(f'Generating Majorana operators for L={args.L} ...')
    majorana_list = gen_maj(args.LA)
    logging.info(f"Memory usage after building Majorana: \
        {process.memory_info().rss / 1024 / 1024} MB")
    
    logging.info(f'Building GA for L={args.L}, LA={args.LA}, \
                    seed={args.seed} ...')
    rng = np.random.default_rng(args.seed)
    cpls = rng.normal(size=comb(2*args.LA, 4, exact=True))
    iterator = np.asarray(
        [list(comb) for comb in combinations(range(2*args.LA), 4)]
    )
    GA = build_GA(majorana_list, args.L, args.LA, cpls, iterator)
    logging.info(f'GA dim: {GA.shape}')
    logging.info(f"Memory usage after building GA: \
        {process.memory_info().rss / 1024 / 1024} MB")

    if args.save:
        logging.info(f'Saving GA for L={args.L}, LA={args.LA}, \
                        seed={args.seed} ...')
        GA = cp.asnumpy(GA)
        np.save(os.path.join(
            args.save_dir, \
                f'GA_L={args.L}_LA={args.LA}_seed={args.seed}.npy'), GA)
        logging.info(f"Memory usage after saving GA: \
            {process.memory_info().rss / 1024 / 1024} MB")
    
if __name__ == '__main__':
    pipeline()
