import numpy as np
from scipy.special import comb
from itertools import combinations
from sys import argv
import json

rng = np.random.default_rng(seed=0)

L = int(argv[1])
N = 2*L
maj_combs = {key: 0. for key in list(combinations(range(N),4))}
idxs_cpl = {}
for idxs in combinations(range(N),4):
    idxs_cpl[idxs] = rng.normal()

def coeff(key):
    def reversed_num(lst):
        num = 0
        for i in range(len(lst)-1):
            for j in range(i+1,len(lst)):
               if lst[i] > lst[j]:
                   num += 1
        return num
    
    idxs_rem = [idx for idx in range(N) if idx not in key]
    idxs_com_all = list(combinations(idxs_rem, 2))
    key_divide = [(combo, tuple(set(key) - set(combo))) for combo in combinations(key, 2)]
    coeff = 0.
    
    for k, idxs_com in enumerate(idxs_com_all):
        for key1, key2 in key_divide:
            if k == 0:
                print(key1, key2)
            lst1 = list(key1) + list(idxs_com)
            lst2 = list(idxs_com) + list(key2)
            reversed_num_key_divide = reversed_num(list(key1) + list(key2))
            reversed_num_1 = reversed_num(lst1)
            reversed_num_2 = reversed_num(lst2)
            sgn = -1 if (reversed_num_key_divide + reversed_num_1 + reversed_num_2) % 2 == 0 else 1
            contribution = sgn*idxs_cpl[tuple(sorted(lst1))]*idxs_cpl[tuple(sorted(lst2))]
            coeff += contribution
            obj[str(key1)+','+str(idxs_com)+','+str(key2)] = contribution
    return coeff

obj = {}
_ = coeff((1,2,5,7))
with open('look_into_sgns.json','w') as f:
    json.dump(obj,f,indent=4)
