import dynamite
from dynamite.operators import Operator
from dynamite.computations import dm_entanglement_entropy
import numpy as np
from scipy.sparse.linalg import expm, svds
import os
import resource


msc_dir = '/n/home01/ytan/scratch/deviation_ee/msc_GA'
GA = Operator.load(os.path.join(msc_dir, 'GA_LA=6_seed=0.msc'))
GA_numpy = GA.to_numpy()

usage = resource.getrusage(resource.RUSAGE_SELF)
print(f"Memory usage: {usage.ru_maxrss / 1024} MB")

tr = (GA_numpy @ GA_numpy).trace()
print(tr/2**6/2**9)
for beta in np.linspace(-0.1, 0.1, 11):
    dm = expm(-beta*GA_numpy)
    trace = dm.trace()
    dm /= trace
    # _, S, _ = svds(dm)
    dm = dm.toarray()
    ee = dm_entanglement_entropy(dm)
    print(f'beta={beta}, ee={ee}')

