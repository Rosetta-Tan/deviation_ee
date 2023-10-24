import argparse
from dynamite.operators import sigmax, sigmay, sigmaz, identity, op_sum, op_product, index_sum
from dynamite.extras import majorana
from dynamite.states import State
from dynamite.subspaces import Explicit, Full
from dynamite.computations import eigsolve, entanglement_entropy
from dynamite import config
import numpy as np
import os
from timeit import default_timer as timer
from itertools import combinations

def ham_sp_case():
    '''Special case H = sum_i sigma^z_i
    '''
    ham = index_sum(sigmaz(), boundary='open')
    return ham

def ham_mixfld_ising(J=1.0, hx=0.2, hz=0.2):
    '''Build a local chaotic Hamiltonian (spin-chain).
    H = \sum_i \sigma_i^x \sigma_{i+1}^x
    
    Parameters:
    g: some parameter
    h: another parameter
    
    Returns:
    h: Hamiltonian
    '''
    ham_zz = index_sum(sigmaz(0) * sigmaz(1), boundary='open')
    ham_x = index_sum(hx * sigmax())
    ham_z = index_sum(hz * sigmaz())
    ham = op_sum([ham_zz, ham_x, ham_z])
    return ham

def ham_syk(J=1.0):
    '''Build SYK Hamiltonian.
    H = \sum_{i<j<k<l} J_{ijkl} \sigma_i \sigma_j \sigma_k \sigma_l

    Parameters:
    
    '''
    rng = np.random.default_rng(seed=42)
    # only compute the majoranas once
    majoranas = [majorana(i) for i in range(2*config.L)]

    def gen_products():
        for idxs in combinations(range(2 * config.L), 4):
            p = op_product(majoranas[idx] for idx in idxs)
            p.scale(np.sqrt(6/(config.L**3)) * rng.normal(loc=0, scale=J))
            yield p

    H = op_sum(gen_products(config.L))
    return H

def main():
    # random sample a state |v> from the whole Hilbert space
    # |psi_i> e^-{H^2/2}|v_i> / || e^-{H^2/2}|v_i> ||
    # calculate entanglement entropy S_A^(i) of |psi_i>
    # average over random samples to get S_A
    
    parser = argparse.ArgumentParser(description='Deviation from maximal entropy.')
    parser.add_argument('--L', dest='L', type=int, required=True, help='system size')
    parser.add_argument('--seed', '-s', dest='seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--shell', default=False, action='store_true')
    args = parser.parse_args()
    
    if args.gpu:
        config.initialize(gpu=True, slepc_args=['-bv_type', 'vecs'])
    if args.shell:
        config.shell = True
        config.initialize(slepc_args=['-bv_type', 'vecs'])
    else:
        config.initialize(slepc_args=['-bv_type', 'vecs'])
    from petsc4py import PETSc
    L = args.L
    config.L = L
    seed = args.seed
    beta = 10
    datadir = "/n/home01/ytan/scratch/deviation-max-ee/data/05-08-2023-mixfld_ising_dnm_imagevol"
    outdir = "/n/home01/ytan/scratch/deviation-max-ee/out/05-08-2023-mixfld_ising_dnm_imagevol"
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    datafile = os.path.join(datadir, f'L{L}_beta{beta}_seed{seed}.npz')

    h = ham_mixfld_ising()
    weht_op = h * h
    
    tic = timer()
    # random sampling from full Hilbert space
    v = State(state='random', seed=seed)
    # imaginary time evolution
    psi = weht_op.evolve(v, t=-0.5j * beta)
    # normalize the state
    psi.normalize()
    # calculate the half system entanglement entropy
    ent_num = entanglement_entropy(psi, np.arange(config.L // 2))
    ent_th = config.L / 2 * np.log(2) - 0.5
    diff = ent_th - ent_num
    toc = timer()
    time = toc - tic
    if PETSc.COMM_WORLD.rank == 0:
        np.savez(datafile, time=time, ent_num=ent_num, ent_th=ent_th, diff=diff)
        print(f'ent_num: {ent_num}, ent_th: {ent_th}, diff: {diff}, time: {toc - tic}')
        print("The end")
    
if __name__ == '__main__':
    main()