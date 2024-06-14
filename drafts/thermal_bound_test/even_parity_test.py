from dynamite.operators import Operator
from dynamite.subspaces import Parity
from dynamite.extras import majorana
from dynamite.operators import op_sum, op_product
from dynamite import config
import numpy as np
from scipy.special import comb
import os
import unittest
from build_GA_numpy import gen_maj

"""
Dynamite routine:
Operator.load(...) -> set subspace to be even

numpy routine:
np.load() -> projector @ GA @ projector.T

compare the two resulting matrices
"""

class TestEvenParity():
    def __init__(self):
        self.L = 12
        self.LA = 6
        self.seed = 0
        self.op_dir = '/n/home01/ytan/scratch/deviation_ee/msc_npy_GA'

    def get_GA_dnm(self):
        """
        Get GA from dynamite
        """
        filename = f'GA_L={self.L}_LA={self.LA}_seed={self.seed}_dnm.msc'
        GA = Operator.load(os.path.join(self.op_dir, filename))
        # GA.add_subspace(Parity('even'))
        GA_numpy = GA.to_numpy(sparse=False)
        normalization_factor = 1.0 / comb(2*self.L, 4)
        return GA_numpy * normalization_factor

    def get_GA_cuda(self):
        """
        Get GA from cupy
        """
        filename = f'GA_L={self.L}_LA={self.LA}_seed={self.seed}.npy'
        GA = np.load(os.path.join(self.op_dir, filename))
        projector = np.zeros((2**(self.LA-1), 2**self.LA), dtype=int)
        for i in range(2**self.LA):
            basis_vec = np.binary_repr(i, width=self.LA)
            if basis_vec.count('1') % 2 == 0:
                projector[i//2, i] = 1
        # GA = projector @ GA @ projector.T
        return GA
    
    def get_GA_np(self):
        """
        Get GA from numpy
        """
        filename = f'GA_L={self.L}_LA={self.LA}_seed={self.seed}_purenp.npy'
        GA = np.load(os.path.join(self.op_dir, filename))
        # projector = np.zeros((2**(self.LA-1), 2**self.LA), dtype=int)
        # for i in range(2**self.LA):
        #     basis_vec = np.binary_repr(i, width=self.LA)
        #     if basis_vec.count('1') % 2 == 0:
        #         projector[i//2, i] = 1
        # GA = projector @ GA @ projector.T
        return GA
    
    def test_majorana_simple(self):
        """
        By doing this test, I realized dynamite is using
        big-endian convention.
        e.g., 0b0001 -> 1, 0b0010 -> 2, 0b0100 -> 4, 0b1000 -> 8
        This means the first Majorana operator is
        M_0 = \sigma_0 \otimes \sigma_x,
        and the second is
        M_1 = \sigma_0 \otimes \sigma_y.
        Instead, small-endian is like
        e.g., 0b0001 -> 8, 0b0010 -> 4, 0b0100 -> 2, 0b1000 -> 1
        """
        config.L = 2
        sigmax = np.array([[0, 1.], [1., 0]], dtype=complex)
        sigmay = np.array([[0, -1.j], [1.j, 0]], dtype=complex)
        sigmaz = np.array([[1., 0], [0, -1.]], dtype=complex)
        sigma0 = np.array([[1., 0], [0, 1.]], dtype=complex)
        majorana_dnm = [majorana(idx) for idx in range(0, 2*config.L)]
        majorana_0 = np.kron(sigma0, sigmax)
        assert(np.allclose(majorana_dnm[0].to_numpy(sparse=False), majorana_0))

    def test_majorana(self):
        """
        Compare the Majorana operators
        """
        config.L = self.LA
        majorana_dnm = [majorana(idx) for idx in range(0, 2*self.LA)]
        majorana_np = gen_maj(self.LA)
        for i in range(0, 2*self.LA):
            majorana_dnm_np = majorana_dnm[i].to_numpy(sparse=False)
            # print(f'majorana_dnm[{i}] - majorana_np[{i}] = {majorana_dnm[i] - majorana_np[i]}')
            assert(np.allclose(majorana_dnm_np, \
                               majorana_np[i], atol=1e-5)), \
                f'majorana_dnm[{i}] and majorana_np[{i}] are inconsistent; \
                majorana_dnm[{i}] - majorana_np[{i}] = {majorana_dnm_np - majorana_np[i]}'
        print('Majorana operators are consistent')

    def test_even_parity(self):
        GA_dnm = self.get_GA_dnm()
        GA_cuda = self.get_GA_cuda()
        GA_np = self.get_GA_np()
        print(f'GA_dnm - GA_np = {GA_dnm - GA_np}')
        assert(np.allclose(GA_dnm-GA_np, np.zeros_like(GA_dnm), atol=1e-5))
        print('GA_dnm and GA_np are consistent')

if __name__ == '__main__':
    test = TestEvenParity()
    test.test_majorana_simple()
    test.test_majorana()
    test.test_even_parity()