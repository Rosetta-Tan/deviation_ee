from dynamite import config
from dynamite.operators import op_sum, op_product, Operator
from dynamite.extras import majorana
from itertools import combinations
import numpy as np
import argparse, os

from timeit import default_timer
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, required=True, help='system size')
parser.add_argument('-J', type=float, required=False, default=1.0, help='coupling strength')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--dirc', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk',  help='msc file directory')
args = parser.parse_args()
L = args.L
J = args.J
seed = args.seed
msc_dir = args.dirc
if not os.path.isdir(msc_dir):
    os.mkdir(msc_dir)

config.L = L
N = 2*L # number of Majoranas

start = default_timer()

# load the cleverly decomposed H2 (without trace)
H2_6uniq = Operator.load(os.path.join(msc_dir,'H2_6uniq_'+str(L)+'_'+str(seed)+'.msc'))
H2_8uniq = Operator.load(os.path.join(msc_dir,'H2_8uniq_'+str(L)+'_'+str(seed)+'.msc'))
H2 = (J**2)*24/(N*(N-1)*(N-2)*(N-3))*op_sum([H2_6uniq, H2_8uniq])
H2_8uniq.save(os.path.join(msc_dir,f'H2_{L}_{seed}.msc'))

print(f'assemble h2, L={L}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}')