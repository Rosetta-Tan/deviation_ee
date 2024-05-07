import csv, os
import numpy as np

Ls = list(range(10, 31))
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

        # with open(filename, 'r') as file:
        #     # skip the first line
        #     next(file)
        #     reader = csv.reader(file)
        #     for row in reader:
        #         # if the row is not a comment
        #         if len(row) > 0 and not row[0].startswith('#'):
        #             if int(row[0]) == L:
        #                 ees.append(float(row[2]))
        #                 break