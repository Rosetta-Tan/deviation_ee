import dynamite
from dynamite.states import State
from dynamite.computations import reduced_density_matrix
from dynamite.operators import Operator, zero, op_product, op_sum
from dynamite.extras import majorana
from dynamite import config
from dynamite.subspaces import Parity
import os
import unittest
import numpy as np
from scipy.special import comb
from itertools import combinations
from timeit import default_timer

vec_dir = '/n/home01/ytan/scratch/deviation_ee/vec_syk_pm_z2_newtol'
msc_dir = '/n/home01/ytan/scratch/deviation_ee/msc_GA'

class TestProjectStateVec(unittest.TestCase):
    def test_project_state_vec(self):
        """
        This test is to show that the metadata L of the state vector is just L, not L//2.
        Not like Operator, State recorded Parity symmetry, the stored dimension being half of the actual dimension.
        But as this test shows, the metadata L is not changed.
        """
        L = 20
        LA = L // 2
        seed = 0
        tol = 0.001
        filename_str = f'v_L={L}_seed={seed}_tol={tol}'
        v = State.from_file(os.path.join(vec_dir, filename_str))
        self.assertEqual(v.L, L)

    def test_reduced_density_matrix(self):
        """
        This test is to show that the extension of the operator GA by setting GA.L=L is truly just GA tensor IAbar,
        by matching the expectation value taken with v.
        """
        seed = 0
        tol = 0.001
        L = 12
        N = 2 * L
        normalization_factor = 1.0 / comb(N, 4)
        LA = L // 2
        NA = 2 * LA
        A_maj_inds = list(range(NA))
        iterator = list(combinations(A_maj_inds, 4))
        RNG = np.random.default_rng(seed)
        M = [majorana(idx) for idx in range(N)]
        CPLS = RNG.normal(size=comb(N, 4, exact=True))
        CPLS_MAP = {(i,j,k,l): CPLS[ind] for ind, (i,j,k,l) in enumerate(combinations(range(N), 4))}

        print('building GA_full')
        coeffs = []
        ops = []
        # GA_full.L = L
        for inds1 in iterator:
            for inds2 in iterator:
                cdnl = len(set(inds1).union(set(inds2)))  # cardinality
                if cdnl == 6 or cdnl == 8:
                    C_cdnl = comb(N, NA) / comb(N-cdnl, NA-cdnl)
                    coeffs.append(C_cdnl * CPLS_MAP[inds1] * CPLS_MAP[inds2])
                    ops.append(op_product([M[i] for i in inds1] + [M[j] for j in inds2]))
        GA_full = op_sum([c*op for c, op in zip(coeffs, ops)])
        GA_full.L = L
        GA_full *= normalization_factor
        GA_full.add_subspace(Parity('even'))
        print(GA_full.dim)

        print('loading v')
        v = State.from_file(os.path.join(vec_dir, f'v_L={L}_seed={seed}_tol={tol}'))
        print(v.to_numpy().shape)
        expt_full = v.dot(GA_full.dot(v)).real
        
        print('loading GA')
        GA = Operator.load(os.path.join(msc_dir, f'GA_L={L}_LA={LA}_seed={seed}.msc')) * (1/2)**4 * normalization_factor
        rdm: np.ndarray = reduced_density_matrix(v, range(LA))
        GA_numpy = GA.to_numpy(sparse=False)
        expt = np.trace(rdm @ GA_numpy).real

        self.assertAlmostEqual(expt_full, expt)

def expand_GA():
    L = 12
    LA = L // 2
    seed = 0
    tol = 0.001
    filename_str = f'/n/home01/ytan/scratch/deviation_ee/msc_GA/GA_L={L}_LA={LA}_seed={seed}.msc'
    GA = Operator.load(filename_str)
    GA.L = L
    GA.add_subspace(Parity('even'))
    
    v = State.from_file(os.path.join(vec_dir, f'v_L={L}_seed={seed}_tol={tol}'))
    
    return v.dot(GA.dot(v)).real

def ruduce_v():
    L = 12
    LA = L // 2
    seed = 0
    tol = 0.001
    filename_str = f'/n/home01/ytan/scratch/deviation_ee/msc_GA/GA_L={L}_LA={LA}_seed={seed}.msc'
    GA = Operator.load(filename_str)
    GA_numpy = GA.to_numpy()
    
    v = State.from_file(os.path.join(vec_dir, f'v_L={L}_seed={seed}_tol={tol}'))
    rdm: np.ndarray = reduced_density_matrix(v, range(LA))
    
    return np.trace(rdm @ GA_numpy).real


if __name__ == '__main__':
    # unittest.main()
    """
    This test is to show which approach of evaluating <v|GA|v> is faster.
    """
    start = default_timer()
    expt1 = expand_GA()
    end = default_timer()
    print(f'expand_GA: {expt1}, time: {end-start}')
    start = default_timer()
    expt2 = ruduce_v()
    end = default_timer()
    print(f'ruduce_v: {expt2}, time: {end-start}')
    