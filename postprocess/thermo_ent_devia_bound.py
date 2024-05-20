import csv, os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
plt.style.use('../figures/norm.mplstyle')
from scipy.optimize import curve_fit

Ls = [12, 14, 16, 18]
nseeds = 20
tols = [0.001]
obs_dir = '/n/home01/ytan/scratch/deviation_ee/obs_syk'

def calc_deviation(L, seed, tol):
    S_thermal_0 = L//2 * np.log(2)
    with open(os.path.join(obs_dir, f'thermal_entropy_L={L}_seed={seed}_tol={tol}.csv'), 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            if int(row[1]) == seed:
                if len(row) == 3:
                    S_thermal_beta = complex(row[2]).real
                else:
                    S_thermal_beta = complex(row[3]).real
    return S_thermal_0 - S_thermal_beta

if __name__ == '__main__':
    deviations = np.zeros((len(Ls), len(tols), nseeds))
    for i, L in enumerate(Ls):
        for j, tol in enumerate(tols):
            for seed in range(nseeds):
                deviations[i, j, seed] = calc_deviation(L, seed, tol)
    print(deviations)

    # fit the deviations for each tol using formula a*L^4

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for j, tol in enumerate(tols):
        mean = np.mean(deviations[:, j, :], axis=-1)
        std = np.std(deviations[:, j, :], axis=-1)
        ax.errorbar(Ls, mean, yerr=std, fmt='o-', label=f'tol={tol}')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\Delta S$')
    ax.legend()
    fig.tight_layout()

    fig.savefig('../figures/thermo_ent_devia_bound.pdf')

