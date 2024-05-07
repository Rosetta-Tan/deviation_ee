import argparse 
import numpy as np
from time import time
from scipy.special import comb
from mpi4py import MPI

from dynamite import config
from dynamite.subspaces import SpinConserve
#from dynamite.subspaces import *
from dynamite.operators import sigmax, sigmay, sigmaz, op_sum, index_product
from dynamite.msc_tools import *
from dynamite.tools import *


# Run the Heisenberg model on the Kagome lattice
# Requires the number of spins and the number of up spins

# Saves the eigenvector as a binary file
def main():
    parser = argparse.ArgumentParser(description="Run Heisenberg on the kagome lattice.")
    # Set the default for the dataset argument
    parser.add_argument('--N', dest='N', default="12", type=str)
    parser.add_argument('--num_up_spins', dest='U', default=6, type=int)
    parser.add_argument('--shell', default=False, action='store_true')
    parser.add_argument('--spinflip', default='None', type=str)
    parser.add_argument('--monitor', default=False, action='store_true')
    args = parser.parse_args()

    N = int(''.join(i for i in args.N if i.isdigit()))
    num_up = args.U

    # Load the edges and the ground state energy
    dir_path = "/n/home06/pkrastev/rc_team/tests/dynamite_test/kagome/"
    edges_file_path = dir_path + "data/edges%s.txt"%(args.N)
    #gs_file_path = dir_path + "data/gs%s.txt"%(args.N)

    edges = np.loadtxt(edges_file_path, dtype=int)
    #gs = np.loadtxt(gs_file_path, dtype=float)

    # Whether to print iteration info using the PETSc monitor
    if args.monitor:
        config.initialize(['-eps_monitor'])#, '-eps_view', '-log_view'])
    
    # Parse spinflip option
    spinflip = None
    if args.spinflip == "+" or args.spinflip == "-":
        spinflip = args.spinflip
    elif args.spinflip != "None":
        raise ValueError("spinflip is either + or -")
    # Create hamiltonian
    config.L = int(N)
    h = 1/4*op_sum(sigmax(i)*sigmax(j) + \
               sigmay(i)*sigmay(j) + \
               sigmaz(i)*sigmaz(j) for i, j in edges)

    # Whether to use shell mode
    h.shell = args.shell
    tic = time()
    subspace = SpinConserve(config.L, num_up, spinflip=spinflip)
    toc = time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print("Subspace (s): ", toc - tic, flush=True)

    h.subspace = subspace
    
    #before = get_cur_memory_usage()
    tic = time()
    #energies, eigvecs = h.eigsolve(getvecs=True)
    energies = h.eigsolve()
    toc = time()
    #after = get_cur_memory_usage()
    '''
    from petsc4py import PETSc
    
    # Save the eigenvector
    eigvec_file_path = dir_path + "heis_%s_vec.dat"%(args.N)
    w_viewer = PETSc.Viewer().createBinary(eigvec_file_path, 'w',
               comm=PETSc.COMM_WORLD)
    eigvecs[0].vec.view(w_viewer)

    # Check 
    r_viewer = PETSc.Viewer().createBinary(eigvec_file_path, 'r',
               comm=PETSc.COMM_WORLD)
    u = PETSc.Vec().load(r_viewer)
    assert eigvecs[0].vec.equal(u)
    '''
    if rank == 0:
        print("Eigsolve (s): ", toc - tic)
        print("lowest energy: %f" %(energies[0]))
        #print("expected energy: %f" %(gs))
        #print("relative error: %.2e" %(np.abs((energies[0] - gs)/gs)))
        #print('matrix memory usage: %f Mb' % ((after-before)/1E6))
  
if __name__ == "__main__":
    main()