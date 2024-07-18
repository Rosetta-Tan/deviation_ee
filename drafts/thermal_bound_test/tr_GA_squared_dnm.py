import os
import argparse
import logging
from timeit import default_timer
import numpy as np
from scipy.special import comb
from itertools import combinations

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[FlushStreamHandler()])

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

def load_GA(L, seed):
    '''
    Return:
        GA: ndarray
    '''
    GA_dir = '/n/home01/ytan/scratch/deviation_ee/msc_npy_GA'
    LA = L // 2
    filename = os.path.join(GA_dir, f'GA_L={L}_LA={LA}_seed={seed}_dnm_decmp.npy')
    GA = np.load(filename)
    return GA

def calc_tr_GA_squared(GA):
    """
    Calculate the trace of GA^2
    Params
        GA: array_like
    """
    GA_np = GA
    return np.trace(GA_np @ GA_np)
      
def pipeline():
    L = args.L
    LA = L // 2
    # filename = f'dnm_data_NA={2*args.LA}.csv'
    # start = default_timer()
    # GA = build_GA(list(range(args.LA)))
    # print(GA.to_numpy().shape)
    # tr = calc_tr_GA_squared(GA)
    # end = default_timer()
    # print(f'time elapsed: {end-start}')
    # print(f'tr_GA_squared = {tr/(2**args.LA)/2**9}')
    # if not os.path.exists(filename):
    #     with open(filename, 'w') as f:
    #         f.write('NA,seed,tr_GA_squared\n')
    # with open (filename, 'a') as f:
    #     f.write(f'{2*args.LA}, {args.seed}, {tr/(2**args.LA)/2**9}\n')
    logging.info(f"Loading GA, L={L}, seed={args.seed}")
    GA = load_GA(args.L, args.seed)
    logging.info(f"Calculating tr_GA^2, L={L}, seed={args.seed}")
    tr = calc_tr_GA_squared(GA)
    print(f'tr_GA_squared = {tr/(2**LA)/2**9}')
    if not os.path.exists(f'data_dnm_L={L}.csv'):
        with open (f'data_dnm_L={L}.csv', 'w') as f:
            f.write('NA,seed,tr_GA_squared\n')
    with open (f'data_dnm_L={L}.csv', 'a') as f:
        f.write(f'{L}, {args.seed}, {tr/(2**LA)/2**9}\n')

if __name__ == '__main__':
    pipeline()



