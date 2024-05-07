import os

Ls = list(range(21,25))
with open(f'./extrem_eigvals_fail_count.txt', 'w') as f:
    # clear the file
    f.write('')
    for L in Ls:
        for seed in range(0, 100):
            if not os.path.isfile(f'/n/home01/ytan/scratch/deviation_ee/output/solve_extrm_eigvals/evals_L={L}_seed={seed}.npz'):
                f.write(f'L={L}, seed={seed} not done\n')
        