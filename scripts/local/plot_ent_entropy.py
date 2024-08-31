# usage: python plot_ent_entropy.py

import csv, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.style.use('../../figures/norm.mplstyle')

# Ls = list(range(12, 27, 2))
Ls = [12]
seeds = list(range(20))
tols = [0.1, 0.01, 0.001]
obs_dir = '../../data/obs_syk'

# get simulation data
# ees: {tol1: [[seed1, seed2, ...], ...], tol2: [[seed1, ...], ...], ...}
ees = {}
for tol in tols:
    ees[tol] = []
    for L in Ls:
        ees[tol].append([])
        for seed in seeds:
            with open(os.path.join(obs_dir, f'ent_entropy_L={L}_seed={seed}_tol={tol}.csv'), 'r') as file:
                last_line = file.readlines()[-1]
                ees[tol][-1].append(float(last_line.split(',')[2]))
    ees[tol] = np.asarray(ees[tol])
    assert ees[tol].shape == (len(Ls), len(seeds))

# get thermal entropy data
# th_ents: {tol1: [[seed1, seed2, ...], ...], tol2: [[seed1, ...], ...], ...}
th_ents = {}
for tol in tols:
    th_ents[tol] = []
    for L in Ls:
        th_ents[tol].append([])
        for seed in seeds:
            with open(os.path.join(obs_dir, f'thermal_entropy_L={L}_seed={seed}_tol={tol}.csv'), 'r') as file:
                # # content: L, seed, S_thermal
                # reader = csv.reader(file)
                # next(reader)  # skip header
                # th_ents[tol][-1].append(float(row[2]))
                last_line = file.readlines()[-1]
                th_ents[tol][-1].append(float(last_line.split(',')[3]))
    th_ents[tol] = np.asarray(th_ents[tol])
    assert th_ents[tol].shape == (len(Ls), len(seeds))

# get exact page value
ees_page = []
for L in Ls:
    with open(f'../../syk/exact_page_vals.csv', 'r') as file:
        next(file)
        reader = csv.reader(file)
        for row in reader:
            if not row[0].startswith('#'):
                if int(row[0]) == L:
                    ees_page.append(float(row[1]))
                    break
ees_page = np.array(ees_page).reshape(-1, 1).repeat(len(seeds), axis=1)

deltas = {}
for tol in tols:
    # deltas[tol] = np.array(ees_page)-np.average(ees[tol], axis=1)
    deltas[tol] = np.asarray(ees_page) - ees[tol]
    label_width = 38  # Adjust this value based on the longest label
    print(f'{"Average entropy for tol=" + str(tol):<{label_width}}: {np.average(ees[tol], axis=1)}')
    print(f"\033[92m{'Difference for tol=' + str(tol):<{label_width}}: {np.average(deltas[tol], axis=1)}\033[0m")
    print(f'{"Standard deviation for tol=" + str(tol):<{label_width}}: {np.std(deltas[tol], axis=1)}')

deltas_thermal = {}
thermal_bound = {}
for tol in tols:
    maximally_allowed = (np.asarray(Ls)//2) * np.log(2)
    deltas_thermal[tol] = maximally_allowed - np.average(th_ents[tol], axis=1)
    assert deltas_thermal[tol].shape == (len(Ls), )
    label_width = 38  # Adjust this value based on the longest label
    print(f"{'Average thermal entropy for tol=' + str(tol):<{label_width}}: {np.average(th_ents[tol], axis=1)}")
    print(f"\033[92m{'Difference for tol=' + str(tol):<{label_width}}: {deltas_thermal[tol]}\033[0m")

# Generate a color map with increasing brightness
colors = plt.colormaps.get_cmap('PuBu')
bright_colors = [colors(i) for i in [0.3,0.6,0.9]]

fig, ax = plt.subplots()
for i, tol in enumerate(tols):
    ax.errorbar(Ls, np.average(deltas[tol], axis=1), yerr=np.std(deltas[tol], axis=1),
                fmt='-o', capsize=5, label=f'tol={tol}',
                markeredgecolor='black', markeredgewidth=1,
                color=bright_colors[i])
    ax.plot(Ls, deltas_thermal[tol], '-o', color=bright_colors[i])
ax.set_xlabel('L')
ax.set_xticks(Ls)
ax.set_ylabel('$\Delta S$')
ax.legend()
fig.savefig('../../figures/ent_entropy.pdf', bbox_inches='tight')
