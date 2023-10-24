
from os.path import join
import numpy as np
from timeit import default_timer
from datetime import timedelta
import argparse as ap

# get command line arguments. need to do this before initializing dynamite
p = ap.ArgumentParser(description='Compute Entanglement entropy for SYK model.')

p.add_argument('-L',        type=int,   required=True, help='spin chain length')
p.add_argument('-B',        type=float, required=True, help='value of beta for thermal ensemble')
p.add_argument('--start',   type=float, default=0.,    help='start time')
p.add_argument('--stop',    type=float, default=10.,   help='stop time')
p.add_argument('--points',  type=int,   default=21,    help='number of intermediate time points')

p.add_argument('-m',        type=str,   required=True, help='location of MSC files')
p.add_argument('--H_idx',   type=int,   default=0,     help='index of the Hamiltonian to use')
p.add_argument('--seed',    type=int,   default=1,     help='seed for state building PRNG')
p.add_argument('--tol',     type=float, default=1E-7,  help='tolerance of computation')
p.add_argument('--algo',    type=str,   default='krylov',help='algorithm for matrix exponential')
p.add_argument('-s',        action='store_true',       help='whether to use shell matrices')
p.add_argument('--outfile', type=str,   required=True, help='location to save output')
p.add_argument('--slepc_args',type=str, default='',    help='Arguments to pass to SLEPc')

args = p.parse_args()

from dynamite import config
config.L = args.L
config.shell = args.s

init_args = [] + args.slepc_args.split(' ')

# could do this with argparse, but I like it better this way
if '-mfn_ncv' not in init_args:
    init_args += ['-mfn_ncv','5']

config.initialize(init_args)

from petsc4py.PETSc import Sys
Print = Sys.Print

from dynamite.operators import load
from dynamite.states import State
from dynamite.extras import majorana
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.subspace import Parity
track_memory()

class EventTimer:
    def __init__(self):
        self.global_start = default_timer()
        self.sub_start = None

    @classmethod
    def _td(cls,start,end):
        return timedelta(0,end-start)

    def end_event(self,t=None):

        if t is None:
            t = default_timer()

        Print(' (%s, cur %s Gb, max %s Gb)' % (self._td(self.sub_start,t),
                                               str(get_cur_memory_usage()/1E9),
                                               str(get_max_memory_usage()/1E9)),sep='')
        ss = self.sub_start
        self.sub_start = None
        return t - ss

    def begin_event(self,name):
        new_start = default_timer()
        if self.sub_start is not None:
            d = self.end_event(new_start)
        else:
            d = None
        Print(self._td(self.global_start,new_start),
              name,end='')
        self.sub_start = new_start
        return d

# print arguments for posterity
Print(
'''
Parameters
----------
''')

for k,v in vars(args).items():
    Print(k,v,sep='\t')

Print()

et = EventTimer()
et.begin_event('Start')

N = 2*args.L # number of Majoranas

et.begin_event('Vec/mfn allocation')

W01 = 1/np.sqrt(2) * majorana(1)
W10 = W01.copy()

W01.right_subspace = Parity("even")
W01.left_subspace = Parity("odd")

W10.right_subspace = Parity("odd")
W10.left_subspace = Parity("even")

V01 = 1/np.sqrt(2) * majorana(0)
V01.right_subspace = Parity("even")
V01.left_subspace = Parity("odd")

# to nicely account for t=0
def general_evolve(H,a,b,t):
    if t == 0:
        a.copy(b)
    else:
        H.evolve(a,t=t,result=b,tol=args.tol,algo=args.algo)

parity_L = config.L-1

# tmp1 is built below
A = State(L=parity_L)
tmp2 = State(L=parity_L)
tmp3 = State(L=parity_L)

et.begin_event('load Hamiltonian from file')

# keep track of the even and odd subspaces separately
H0 = load(join(args.m,str(args.L)+'_'+str(args.H_idx)+'.msc'))
H0.subspace = Parity("even")

H1 = load(join(args.m,str(args.L)+'_'+str(args.H_idx)+'.msc'))
H1.subspace = Parity("odd")

et.begin_event('PETSc matrix build')
H0.build_mat()
H1.build_mat()

et.begin_event('Random state build')

# smaller size due to subspace. eventually this will be
# built into dynamite
tmp1 = State(L=parity_L,state='random',seed=args.seed)

et.begin_event('Thermal operator evolution')

# apply thermal operator
H0.evolve(tmp1,t=-1j*args.B/2,result=A,tol=args.tol,algo=args.algo)

et.end_event()

# output the norm of A before we normalize it
Print('Thermal state norm:',A.norm())
A.normalize()

et.begin_event('Calculating ')


et.begin_event('Save data')

np.save(args.outfile+'_times.npy',times)
np.save(args.outfile+'_fs.npy',fs)
np.save(args.outfile+'_twopoint.npy',fs)

with open(args.outfile+'.opts','w') as f:
    for k,v in vars(args).items():
        f.write('%s,%s\n' % (str(k),str(v)))

et.begin_event('End\n\n')

Print('Total max memory usage:',get_max_memory_usage()/(1E9),'Gb')
