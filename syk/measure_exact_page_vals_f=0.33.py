import csv, os
import numpy as np

Ls = list(range(12, 27, 3))
filename = 'exact_page_vals_f=0.33.csv'

if not os.path.exists(filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['L', 'ee_page'])

def partial_harmonic_sum(L):
    return np.sum([1/i for i in range(2**(2*(L//3))+1, 2**L+1)])

def update_csv(L, ee_page):
    with open(filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([L, ee_page])

if __name__ == '__main__':
    for L in Ls:
        ee_page = partial_harmonic_sum(L) - 0.5 * (2**(L//3) - 1)/(2**(2*(L//3)))
        update_csv(L, ee_page)