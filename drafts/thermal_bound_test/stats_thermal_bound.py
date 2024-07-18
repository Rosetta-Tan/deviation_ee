import numpy as np
# from dynamite.operators import Operator
from dynamite.computations import dm_entanglement_entropy
import os
from scipy.linalg import expm

def get_msc(L, LA, seed, dir):
    # if not os.path.exists(os.path.join(dir, \
    #     f'GA_L={L}_LA={LA}_seed={seed}.msc')):
    #     print(f'GA_L={L}_LA={LA}_seed={seed}.msc not found')
    # GA = Operator.load(os.path.join(dir, \
    #     f'GA_L={L}_LA={LA}_seed={seed}.msc'))
    # return GA.to_numpy(sparse=False)
    GA = np.load(os.path.join(dir, \
        f'GA_L={L}_LA={LA}_seed={seed}_dnm_decmp.npy'))
    return GA

def get_thermal_ee(beta, GA):
    rdm = expm(-beta*GA)
    rdm = rdm / np.trace(rdm)
    return dm_entanglement_entropy(rdm)

def pipeline(Ls, seeds, betas, dir):
    data_tensor = np.zeros((2, len(Ls), len(seeds), len(betas)))
    # ideal_data_tensor = np.zeros((len(Ls), len(betas)))
    for iL, L in enumerate(Ls):
        LA = L // 2
        for iseed, seed in enumerate(seeds):
            GA = get_msc(L, LA, seed, dir)
            for ibeta, beta in enumerate(betas):
                ee = get_thermal_ee(beta, GA)
                data_tensor[0, iL, iseed, ibeta] = ee
                data_tensor[1, iL, iseed, ibeta] = LA * np.log(2)
    np.save('thermal_ee.npy', data_tensor)
                

if __name__ == '__main__':
    # Ls = np.arange(18, 19, 2)
    Ls = [14]
    # seeds = list(range(0, 1))
    seeds = [0]
    betas = np.linspace(-0.01, 0.01, 11)
    dir = '/n/home01/ytan/scratch/deviation_ee/msc_npy_GA'
    pipeline(Ls, seeds, betas, dir)