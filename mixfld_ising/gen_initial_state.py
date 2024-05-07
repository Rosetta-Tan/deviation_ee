from dynamite.states import State
from dynamite import config
import numpy as np
import json

config.L = 6

dim = 2**config.L

def compute_vec_element(state):
    return np.exp(2*np.pi*1j*(state/dim))
                  
s = State()
s.set_all_by_function(compute_vec_element, vectorize=True)
s_np = s.to_numpy()
print(s_np)