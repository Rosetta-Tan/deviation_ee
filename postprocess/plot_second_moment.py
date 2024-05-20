import csv, os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
plt.style.use('../figures/norm.mplstyle')
from scipy.optimize import curve_fit

Ls = list(range(12, 27, 2))
nseeds = 20
tols = [0.1, 0.01, 0.001]
obs_dir = '/n/home01/ytan/scratch/deviation_ee/obs_syk'

# get simulation data
expt_Hs = {}
for tol in tols:
    expt_Hs[tol] = []
    for L in Ls:
        expt_Hs[tol].append([])
        with open(os.path.join(obs_dir, f'expt_H_L={L}_tol={tol}.csv'), 'r') as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for row in reader:
                expt_Hs[tol][-1].append(float(row[2]))
    expt_Hs[tol] = np.asarray(expt_Hs[tol])


second_moments = {}
for tol in tols:
    second_moments[tol] = expt_Hs[tol]**2
    print(f'second moments for tol={tol}:', np.average(second_moments[tol], axis=1))

# Generate a color map with increasing brightness
colors = plt.cm.get_cmap('PuBu')
bright_colors = [colors(i) for i in [0.3,0.6,0.9]]

fig, ax = plt.subplots()
for i, tol in enumerate(tols):
    ax.plot(Ls, np.average(second_moments[tol], axis=1), '-o',
                label=f'tol={tol}',
                markeredgecolor='black', markeredgewidth=1,
                color=bright_colors[i])
ax.set_xlabel('L')
ax.set_xticks(Ls)
ax.set_ylabel('$\langle \psi |H | \psi \\rangle^2$')
ax.legend()
fig.savefig('../figures/second_moment.pdf', bbox_inches='tight')
