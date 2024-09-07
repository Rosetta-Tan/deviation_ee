import csv, os
import numpy as np

Ls = list(range(26,27))
filename = 'exact_page_vals.csv'

def partial_harmonic_sum(L):
    return np.sum([1/i for i in range(2**((L+1)//2)+1, 2**L+1)])

def update_csv(L, ee_page):
    with open(filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([L, ee_page])

if __name__ == '__main__':
    for L in Ls:
        ee_page = partial_harmonic_sum(L) - (2**(L//2)-1)/(2*2**((L+1)//2))
        update_csv(L, ee_page)