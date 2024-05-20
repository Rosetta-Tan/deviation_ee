import csv
import os
from timeit import default_timer
import argparse


def init_data_point(L, seed, tol_coeff, filepath):
    with open(filepath, 'a') as f:
        print(f'{L},{seed},{tol_coeff},PLACEHOLDER,PLACEHODER', file=f, flush=True)

def update_data_point(L, seed, tol_coeff, num_steps, runtime, filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines
    with open(filepath, 'w') as f:
        print('L,seed,tol_coeff,num_steps,runtime', file=f)
        for line in lines:
            if line.startswith(f'{L},{seed},{tol_coeff},'):
                print(f'{L},{seed},{tol_coeff},{num_steps},{runtime}', file=f, flush=True)
            else:
                print(line, file=f, end='')
    
if __name__ == '__main__':
    ds = list(range(12, 21))
    seeds = [0]
    tol_coeffs = [1e-4]
    filepath = f'/n/home01/ytan/deviation_ee/tests/step_test/data.csv'
    for L in ds:
        for seed in seeds:
            for tol_coeff in tol_coeffs:
                update_data_point(L, seed, tol_coeff, 100, 100, filepath)
