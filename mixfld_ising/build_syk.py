from sys import argv
from math import sqrt
from os.path import join

L = int(argv[1]) # number of spins (not # of majoranas!)
n = int(argv[2]) # number of disorder samples
output_dir = '/n/home01/ytan/scratch/deviation_ee/msc_syk'

from dynamite import config
config.L = L

from dynamite.operators import op_sum, op_product
from dynamite.extras import majorana
from itertools import combinations
import numpy as np

from timeit import default_timer
from datetime import timedelta

N = 2*L # number of Majoranas
J = 1.0

# cache Majoranas to save time
M = [majorana(idx) for idx in range(0,N)]

for j in range(n):

    start = default_timer()

    # initialize random number generator
    np.random.seed(j)

    # build Hamiltonian
    H = sqrt(6/(N**3))*op_sum((J*np.random.randn()*op_product((M[idx] for idx in idxs))
                              for idxs in combinations(range(N),4)))

    H.save(join(output_dir,'H_'+str(L)+'_'+str(j)+'.msc'))
    
    print('Built Hamiltonian',j,'\t',timedelta(0,default_timer()-start))
