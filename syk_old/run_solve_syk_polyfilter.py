import subprocess
import os
from timeit import default_timer as timer

# dir = '/n/home01/ytan/scratch/deviation_ee/output/build_6uniq/'
# dir = '/n/home01/ytan/scratch/deviation_ee/output/build_8uniq/'
dir = '/n/home01/ytan/scratch/deviation_ee/output/solve_polyfilter/'

# Ls = [16, 18, 20]
Ls = [16]
seeds = range(1)
start = timer()
for L in Ls:
    for seed in seeds:
        # non-parallel version
        chmod = ['chmod', '+x', os.path.join(dir,f'L={L}_seed={seed}.sh')]
        command = ['sbatch', os.path.join(dir,f'L={L}_seed={seed}.sh')]
        # parallel version
        # chmod = ['chmod', '+x', os.path.join(dir,f'L={L}_seed={seed}_parallel.sh')]
        # command = ['sbatch', os.path.join(dir,f'L={L}_seed={seed}_parallel.sh')]
        subprocess.run(chmod)
        subprocess.run(command)