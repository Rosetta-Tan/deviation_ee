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

def ham_syk(J=1.0):
    '''Build SYK Hamiltonian.
    H = \sum_{i<j<k<l} J_{ijkl} \sigma_i \sigma_j \sigma_k \sigma_l

    Parameters:
    
    '''
    rng = np.random.default_rng(seed=42)
    # only compute the majoranas once
    majoranas = [majorana(i) for i in range(2*config.L)]

    def gen_products(L):
        for idxs in combinations(range(2 * config.L), 4):
            p = op_product(majoranas[idx] for idx in idxs)
            p.scale(np.sqrt(6/((2*config.L)**3)) * rng.normal(loc=0, scale=J))
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
    # parser.add_argument('--seed', '-s', dest='seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--shell', default=False, action='store_true')
    args = parser.parse_args()

    if args.gpu:
        config.initialize(gpu=True, slepc_args=['-bv_type', 'vecs'])
    if args.shell:
        config.shell = True
    config.L = args.L
    ham = ham_syk()

    # TODO: compute the entanglement entropy vs subsystem fraction, i.e., Page curve
    # S_ent is computed using the eigenstate at zero energy for the original Hamiltonian H
    # To compute Page curve, compute the entanglement entropy at different fractions
    # 

    E_gs, gs = evals, evecs = ham.eigsolve(nev=1,getvecs=True,which='smallest',tol=1E-10)

if __name__ == '__main__':
    main()