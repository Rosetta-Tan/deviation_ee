import csv, os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
plt.style.use('../figures/norm.mplstyle')
from scipy.optimize import curve_fit

Ls = list(range(12, 25, 2))
nseeds = 20
tols = [0.1, 0.01]

# get simulation data
ees = {}
for tol in tols:
    ees[tol] = []
    for L in Ls:
        ees[tol].append([])
        with open(f'ent_entropy_long_tol={tol}.csv', 'r') as file:
            next(file)
            reader = csv.reader(file)
            for row in reader:
                if not row[0].startswith('#'):
                    if int(row[0]) == L:
                        # counting from the current line, read nseeds lines
                        ees[tol][-1].append(float(row[2]))
                        for _ in range(nseeds-1):
                            assert int(row[0]) == L
                            ees[tol][-1].append(float(row[2]))
                            row = next(reader)
                        break
        file.close()

# get exact page value
ees_page = []
for L in Ls:
    with open(f'exact_page_vals.csv', 'r') as file:
        next(file)
        reader = csv.reader(file)
        for row in reader:
            if not row[0].startswith('#'):
                if int(row[0]) == L:
                    ees_page.append(float(row[1]))
                    break

# print('Exact page values: ', ees_page)
# ees = np.asarray(ees)
# print('shape of ees:', ees.shape)
# ees_avg = np.average(ees, axis=1)
# # print('Simulation data:', ees)
# deltas = np.array(ees_page)-np.array(ees_avg)
# print('Difference:', np.array(ees_page)-np.array(ees_avg))
# get average entropy and difference
deltas = {}
for tol in tols:
    deltas[tol] = np.array(ees_page)-np.average(ees[tol], axis=1)
    print(f'Average entropy for tol={tol}:', np.average(ees[tol], axis=1))
    print(f'Difference for tol={tol}:', deltas[tol])

fig, ax = plt.subplots()
for tol in tols:
    ax.errorbar(Ls, deltas[tol], yerr=np.std(ees[tol], axis=1), fmt='-o', capsize=5, label=f'tol={tol}')
ax.set_xlabel('L')
ax.set_ylabel('$\Delta S$')
ax.legend()
fig.savefig('../figures/ent_entropy_page.pdf', bbox_inches='tight')
