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
ees = {}
for tol in tols:
    ees[tol] = []
    for L in Ls:
        ees[tol].append([])
        with open(os.path.join(obs_dir, f'ent_entropy_L={L}_tol={tol}.csv'), 'r') as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for row in reader:
                ees[tol][-1].append(float(row[2]))
    ees[tol] = np.asarray(ees[tol])

# get exact page value
ees_page = []
for L in Ls:
    with open(f'../syk/exact_page_vals.csv', 'r') as file:
        next(file)
        reader = csv.reader(file)
        for row in reader:
            if not row[0].startswith('#'):
                if int(row[0]) == L:
                    ees_page.append(float(row[1]))
                    break
ees_page = np.array(ees_page).reshape(-1, 1).repeat(nseeds, axis=1)

deltas = {}
for tol in tols:
    # deltas[tol] = np.array(ees_page)-np.average(ees[tol], axis=1)
    deltas[tol] = np.array(ees_page) - ees[tol]
    print(f'Average entropy for tol={tol}:', np.average(ees[tol], axis=1))
    print(f'Difference for tol={tol}:', np.average(deltas[tol], axis=1))
    print('Standard deviation for tol={tol}:', np.std(deltas[tol], axis=1))

# Generate a color map with increasing brightness
colors = plt.cm.get_cmap('PuBu')
bright_colors = [colors(i) for i in [0.3,0.6,0.9]]

fig, ax = plt.subplots()
for i, tol in enumerate(tols):
    ax.errorbar(Ls, np.average(deltas[tol], axis=1), yerr=np.std(deltas[tol], axis=1),
                fmt='-o', capsize=5, label=f'tol={tol}',
                markeredgecolor='black', markeredgewidth=1,
                color=bright_colors[i])
ax.set_xlabel('L')
ax.set_xticks(Ls)
ax.set_ylabel('$\Delta S$')
ax.legend()
fig.savefig('../figures/ent_entropy.pdf', bbox_inches='tight')
