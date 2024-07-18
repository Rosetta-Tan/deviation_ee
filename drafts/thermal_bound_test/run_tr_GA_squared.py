import subprocess
import sys, os

Ls = [12, 14, 16]
seeds = list(range(0, 20))

for L in Ls:
    for seed in seeds:
        # plain version
        # command = ['python', 'tr_GA_squared.py', '--LA', str(LA), '--seed', str(seed)]
        # dynamite version
        print(f'progress: L={L}, seed: [{seed}/{len(seeds)}]')
        command = ['python', '/n/home01/ytan/deviation_ee/drafts/thermal_bound_test/tr_GA_squared_dnm.py', '--L', str(L), '--seed', str(seed)]
        subprocess.run(command)