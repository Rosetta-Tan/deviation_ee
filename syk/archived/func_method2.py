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

def gen_terms(key):
    assert len(key) == 6  # this is for the 6-unique element case
    def reversed_num(lst):
        num = 0
        for i in range(len(lst)-1):
            for j in range(i+1,len(lst)):
               if lst[i] > lst[j]:
                   num += 1
        return num
    common_idxs_all = combinations(key, 2)  # choose 2 indices out of 6 to be common indices
    for common_idxs in common_idxs_all:
        rem_idxs = tuple(set(key) - set(common_idxs))
        print(rem_idxs)
        # for key1 in combinations(rem_idxs, 2):
        key1_all = combinations(rem_idxs, 2)  # choose 2 indices out of 4 remaning ones to be in the first copy
        for key1 in key1_all:
            key2 = tuple(set(rem_idxs) - set(key1))  # the remaining 2 indices are in the second copy
            lst1 = list(key1) + list(common_idxs)
            lst2 = list(common_idxs) + list(key2)
            reversed_num_1 = reversed_num(lst1)
            reversed_num_2 = reversed_num(lst2)
            sgn = 1 if (reversed_num_1 + reversed_num_2) % 2 == 0 else -1
            contribution = sgn*idxs_cpl[tuple(sorted(lst1))]*idxs_cpl[tuple(sorted(lst2))]
            # obj[str(key1)+','+str(common_idxs)+','+str(key2)] = (reversed_num_1, reversed_num_2, contribution)
            yield contribution

# key = (1,3,5,6,8,9)
# obj = {}
# lst = list(gen_terms(key))
# with open ('look_into_sgns_method2.json','w') as f:
#     json.dump(obj,f,indent=4)
lst = [gen_terms(key) for key in combinations(range(N),6)]
print(len(lst))