
from os.path import join
import numpy as np
from timeit import default_timer
from datetime import timedelta
import argparse as ap
import json

# get command line arguments. need to do this before initializing dynamite
p = ap.ArgumentParser(description='Compute Entanglement entropy for SYK model.')

p.add_argument('-L',        type=int,   required=True, help='spin chain length')
p.add_argument('-m',        type=str,   required=True, help='location of MSC files')
p.add_argument('--H_idx',   type=int,   default=0,     help='index of the Hamiltonian to use')
p.add_argument('--tol',     type=float, default=1E-7,  help='tolerance of computation')
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

from dynamite.operators import Operator
from dynamite.states import State
from dynamite.extras import majorana
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.subspaces import Parity
from dynamite.computations import entanglement_entropy
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
parity_L = config.L-1
config.subspace = Parity(0)

et.begin_event('load Hamiltonian from file')
start = default_timer()
# keep track of the even and odd subspaces separately
H0 = Operator.load(join(args.m,'H2_'+str(args.L)+'_'+str(args.H_idx)+'.msc'))
H0.subspace = config.subspace
H0.allow_projection = True

# H1 = Operator.load(join(args.m,'H2_'+str(args.L)+'_'+str(args.H_idx)+'.msc'))
# H1.subspace = Parity("odd")
# H0.build_mat()

##############################################
# debug
print('Whether H0 conserves parity', H0.conserves(config.subspace))
print('Whether parity subspace is implemented for H0', H0.has_subspace(config.subspace))
##############################################

et.begin_event('Solving ground state')
evals, evecs = H0.eigsolve(getvecs=True, nev=1, which='smallest', tol=args.tol, subspace=config.subspace)
evec = evecs[0]

et.begin_event('Compute entanglement entropy')
sent = entanglement_entropy(evec,range(config.L//2))

et.begin_event('Save data')
runtime = default_timer() - start
data = {
    'runtime': runtime,
    'sent': sent
}
with open(args.outfile+'.json','w') as f:
    json.dump(data,f,indent=4)
with open(args.outfile+'.opts','w') as f:
    for k,v in vars(args).items():
        f.write('%s,%s\n' % (str(k),str(v)))
et.begin_event('End\n\n')

Print('Total max memory usage:',get_max_memory_usage()/(1E9),'Gb; Runtime:', runtime)
