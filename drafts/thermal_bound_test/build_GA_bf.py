import dynamite
from dynamite.operators import zero, op_product
from dynamite.extras import majorana
from dynamite import config
import numpy as np
import argparse
from itertools import combinations
from scipy.special import comb
import logging
import resource


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[FlushStreamHandler()])

# Function to print current memory usage
def print_memory_usage(msg):
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.info(f"{msg}: {mem_usage / 1024} MB")

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, help='system size')
parser.add_argument('--LA', type=int, help='subsystem size')
parser.add_argument('--seed', type=int, help='random seed')
args = parser.parse_args()
# dynamite configuration
config.L = args.LA
config.initialize(gpu=True)


def build_GA(majorana, LA, cpls, iterator):
    """build subsystem Hamiltonian GA = sum_{J1 J2} C_|J1 \cup J2| \
        K_J1 K_J2 X_J1 X_J2
    """
    GA = np.zeros((2**LA, 2**LA), dtype=np.complex128)
    normalization_factor = 1.0 / comb(2*LA, 4)
    for i1, inds1 in enumerate(iterator):
        for i2, inds2 in enumerate(iterator):
            cdnl = len(set(inds1).union(set(inds2)))
            if cdnl == 6 or cdnl == 8:
                coeff = cpls[i1] * cpls[i2]
                op = op_product([majorana[i] for i in inds1] + \
                                [majorana[i] for i in inds2])
                op = op.to_numpy(sparse=False)
                GA += coeff * op
    return GA * normalization_factor

def pipeline():
    # Set up memory monitoring
    # print_memory_usage("Initial memory usage")

    logging.info(f'Generating Majorana operators for L={args.L} ...')
    majorana_list = [majorana(i) for i in range(2*args.LA)]
    # print_memory_usage("Memory usage after generating Majorana operators")
    
    logging.info(f'Building GA for L={args.L}, LA={args.LA}, \
                    seed={args.seed} ...')
    rng = np.random.default_rng(args.seed)
    cpls = rng.normal(size=comb(2*args.LA, 4, exact=True))
    iterator = list(combinations(range(2*args.LA), 4))
    GA = build_GA(majorana_list, args.LA, cpls, iterator)
    logging.info(f'GA dim: {GA.shape}')
    # print_memory_usage("Memory usage after building GA")
    
if __name__ == '__main__':
    pipeline()