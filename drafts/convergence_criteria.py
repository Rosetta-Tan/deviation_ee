from dynamite import config
from dynamite.states import State
from dynamite.subspaces import Parity
from dynamite.operators import op_sum, op_product, identity, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from dynamite.computations import entanglement_entropy
from itertools import combinations
import numpy as np
from scipy.special import comb
from timeit import default_timer

config.initialize(gpu=True)  # for GPU

def energy_variance(H, psi):
    return H.expt(psi, psi) - H.expt(psi, psi)**2

# We start from randomly sampled states {v0} from the full Hilbert space.
# Evolve v0 with polynomial of H (power method) to get a bunch of {v1}
# For each v1, compute the energy variance


