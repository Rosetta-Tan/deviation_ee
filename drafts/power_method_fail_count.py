import os

Ls = list(range(12, 21))
tol = 0.0001
vec_dir = '/n/home01/ytan/scratch/deviation_ee/vec_syk/powermethod/'
mode = 'w' # choose between 'w' and 'a'

with open(f'./power_method_fail_count.txt', 'w') as f:
    # clear the file
    f.write('')
    for L in Ls:
        for seed in range(0, 100):
            vec_file_bare = os.path.join(vec_dir, f'L={L}_tol={tol}', f'v_L={L}_seed={seed}_state_seed=0_tol_coeff={tol}')
            if not os.path.isfile(f'{vec_file_bare}.metadata') or not os.path.isfile(f'{vec_file_bare}.vec'):
                f.write(f'L={L}, seed={seed} not done\n')
        