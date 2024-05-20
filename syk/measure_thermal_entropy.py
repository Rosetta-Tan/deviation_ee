import os, sys, csv, argparse
from dynamite import config
from dynamite.states import State
from dynamite.operators import zero, op_sum, op_product, identity, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from dynamite.computations import entanglement_entropy, evolve
from dynamite.subspaces import Parity
from itertools import combinations, product
import numpy as np
from scipy.sparse.linalg import expm, svds
from scipy.linalg import logm, svd
from scipy.optimize import root_scalar
from scipy.special import comb
from timeit import default_timer
from datetime import timedelta
import logging

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[FlushStreamHandler()])

parser = argparse.ArgumentParser()
parser.add_argument('--L', required=True, type=int, help='system size')
parser.add_argument('--seed', required=True, type=int, help='random seed')
parser.add_argument('--tol', type=float, help='tolerance for powermethod solving state vector')
parser.add_argument('--log_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/output/20240425_powermethod_z2_newtol', help='directory to save log files')
parser.add_argument('--vec_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/vec_syk_pm_z2_newtol', help='directory to save state vectors')
parser.add_argument('--obs_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/obs_syk', help='directory to save obs files')
parser.add_argument('--gpu', required=False, action='store_true', help='use GPU')
parser.add_argument('--save', type=bool, default=False, help='save data')
args = parser.parse_args()

if args.gpu:
    config.initialize(gpu=True)
config.shell=True  # not using shift-and-invert, so can use shell matrix
config.L = args.L

# cache Majoranas to save time
N = 2*args.L
M = [majorana(idx) for idx in range(0, N)]
RNG = np.random.default_rng(args.seed)
CPLS = RNG.normal(size=comb(N, 4, exact=True))  # the overall constant factor is 1/sqrt((N choose 4))
CPLS_MAP = {(i, j, k, l): CPLS[ind] for ind, (i, j, k, l) in enumerate(combinations(range(N), 4))}  # map from tuple to CPLS index}

def save_data(csv_filepath, *data, append=True):
    with open(csv_filepath, 'a' if append else 'w') as file:
        writer = csv.writer(file)
        writer.writerow([args.L, args.seed, *data])

def build_GA_tensor_IAbar(A_inds: list[int]):
    """
    Build the subsystem Hamiltonian GA with full Hilbert space
    Paras:
        A_inds: indices of *spins* in subsystem A
    """
    GA = zero()
    NA = 2*len(A_inds)
    # this is necessary because the iterator should go over Majorana indices instead of spin indices
    A_maj_inds = list(sorted([2*i for i in A_inds] + [2*i+1 for i in A_inds]))
    normalization_factor = 1.0 / comb(N, 4)
    # print(cpls_subset)
    iterator = combinations(A_maj_inds, 4)
    for inds1 in iterator:  # inds1 is a tuple of 4 indices
        for inds2 in iterator:
            cdnl = len(set(inds1).union(set(inds2)))
            if cdnl == 4 or cdnl == 5 or cdnl == 7:
            # only cardinality 6 or 8 terms are non-zero
                continue
            C_cdnl = comb(N, NA) / comb(N-cdnl, NA-cdnl)
            ops = [M[i] for i in inds1] + [M[j] for j in inds2]
            GA += C_cdnl * CPLS_MAP[inds1] * CPLS_MAP[inds2] * op_product(ops)
    return normalization_factor * GA

def GA_to_numpy(GA: Operator, A_inds: list[int]):
    """
    Convert GA to a sparse CSC matrix.
    Example:
    A_inds = [0, 1, 4, 5]
    then selector = [0, 2^0, 2^1, 2^0+2^1,
                    2^4, 2^0+2^4, 2^1+2^4, 2^0+2^1+2^4,
                    2^5, 2^0+2^5, 2^1+2^5, 2^0+2^1+2^5,
                    2^4+2^5, 2^0+2^4+2^5, 2^1+2^4+2^5, 2^0+2^1+2^4+2^5]
    """
    GA_np = GA.to_numpy()
    selector = np.zeros(2**len(A_inds), dtype=int)
    for i, inds in enumerate(product([0, 1], repeat=len(A_inds))):
        # Assume A_inds is in ascending order, for the selector to also be in ascending order,
        # we need to reverse the order of A_inds
        selector[i] = sum([2**ind for ind, val in zip(reversed(A_inds), inds) if val])
    GA_proj = GA_np[selector, :][:, selector]
    return GA_proj

def test_build_GA_tensor_IAbar(A_inds: list[int]):
    GA = build_GA_tensor_IAbar(A_inds)
    assert GA.dim[0] == 2**(args.L), f'GA dim: {GA.dim}'

def test_GA_to_numpy(A_inds):
    """
    - Test the conversion from GA to numpy array (dimensionality issue)
    - Test the trace of GA^2/2^L_A, to see if it matches the theoretical result
    """
    GA = build_GA_tensor_IAbar(A_inds)
    GA_np = GA.to_numpy()
    # print('(debug) trace of GA^2/2^L_A: ', (GA_np@GA_np).trace()/2**(len(A_inds)))
    assert GA_np.shape == (2**(args.L), 2**(args.L)), f'GA shape: {GA_np.shape}'

def test_GA_to_numpy_projection(A_inds):
    GA = build_GA_tensor_IAbar(A_inds)
    GA_np = GA.to_numpy()
    selector = np.zeros(2**len(A_inds), dtype=int)
    for i, inds in enumerate(product([0, 1], repeat=len(A_inds))):
        selector[i] = sum([2**ind for ind, val in zip(reversed(A_inds), inds) if val])
    print(selector)
    # GA is CSC matrix
    GA_proj = GA_np[selector, :][:, selector]
    assert GA_proj.shape == (2**len(A_inds), 2**len(A_inds)), f'GA_proj shape: {GA_proj.shape}'

def test_GA_dm(beta):
    A_inds = list(range(0, config.L//2))
    GA = build_GA_tensor_IAbar(A_inds)
    GA_numpy = GA_to_numpy(GA, A_inds)
    GA_dm = expm(-beta * GA_numpy)
    assert GA_dm.shape == (2**len(A_inds), 2**len(A_inds)), f'GA_dm shape: {GA_dm.shape}'

def test_thermal_energy(beta):
    # TODO
    pass

def test_thermal_entropy(beta, A_inds):
    GA = build_GA_tensor_IAbar(A_inds)
    GA_numpy = GA_to_numpy(GA, A_inds)
    GA_dm = expm(-beta * GA_numpy).toarray()
    print('rho shape: ', rho.shape)
    Z = np.trace(GA_dm)
    rho = GA_dm / Z
    S_thermal = -np.trace(rho @ logm(rho)).real
    print(f'(debug): beta={beta}, S_thermal={S_thermal}')
    return S_thermal


def get_state_vec():
    filename_str = f'v_L={args.L}_seed={args.seed}_tol={args.tol}'
    try:
        v = State.from_file(os.path.join(args.vec_dir, filename_str))
        logging.info(f'load state vec: L={args.L}, seed {args.seed} ...')
        return v
    except:
        logging.error(f'state vec file not found: L={args.L}, seed {args.seed} ...')
        return None

def prepare_data_for_diff_GA_expt_thermal(A_inds: list[int]):
    """
    Prepare the data for diff_GA_expt_thermal,
    so that in root finding, these calculations are not repeated.
    """
    v = get_state_vec()
    A_inds = list(range(0, config.L//2))
    GA = build_GA_tensor_IAbar(A_inds)
    GA_numpy = GA_to_numpy(GA, A_inds)
    
    # convert GA to numpy before setting the Parity('even') subspace, otherwise will be complicated to project GA to the subsystem
    GA.subspace = Parity('even')
    expt_GA = v.dot(GA.dot(v)).real
    logging.info(f'measure GA expectation value: L={args.L}, seed {args.seed} ...')
    
    return v, expt_GA, GA_numpy

def diff_GA_expt_thermal(v, expt_GA, GA_numpy, beta):
    """
    Diff two results.
    1. Measure the subsystem Hamiltonian GA using state vector |psi>.
    i.e., evaluating <psi| GA |psi> 
    2. Compute the thermal energy of GA,
    i.e., evaluating tr(GA * rho) where rho = exp(-beta * GA) / Z
    """
    GA_dm = expm(-beta * GA_numpy).real
    Z = GA_dm.trace()
    thermal_energy = (GA_dm @ GA_numpy).trace() / Z
    return expt_GA - thermal_energy

def root_find(A_inds: list[int]):
    """
    Find the beta that gives the correct thermal energy.
    Using bisect method.
    i.e., solving tr(GA * rho(beta)) = <psi| GA |psi> for beta
    """
    beta0, beta1 = -100, 100
    v, expt_GA, GA_numpy = prepare_data_for_diff_GA_expt_thermal(A_inds)
    print('at beta0: ', diff_GA_expt_thermal(v, expt_GA, GA_numpy, beta0))
    print('at beta1: ', diff_GA_expt_thermal(v, expt_GA, GA_numpy, beta1))
    sol = root_scalar(lambda beta: diff_GA_expt_thermal(v, expt_GA, GA_numpy, beta), bracket=[beta0, beta1])
    return sol.root, sol.iterations
    
def thermal_entropy(beta):
    """
    Calculate the thermal bound Delta S = S_vN(rho_GA(beta)), at the calculated beta.
    Here S_vN is the von-Neumann entropy.
    """
    A_inds = list(range(0, config.L//2))
    GA = build_GA_tensor_IAbar(A_inds)
    GA_numpy = GA_to_numpy(GA, A_inds)
    GA_dm = expm(-beta * GA_numpy).toarray()
    Z = np.trace(GA_dm)
    rho = GA_dm / Z
    S_thermal = -np.trace(rho @ logm(rho)).real

    if args.save:
        csv_filepath = os.path.join(args.obs_dir, f"thermal_entropy_L={args.L}_seed={args.seed}_tol={args.tol}.csv")
        if not os.path.exists(csv_filepath):
            with open(csv_filepath, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','beta','S_thermal'])
        save_data(csv_filepath, beta, S_thermal, append=True)
    return S_thermal

if __name__ == '__main__':
    A_inds = list(range(0, config.L//2))
    v, expt_GA, GA_numpy = prepare_data_for_diff_GA_expt_thermal(A_inds)
    beta, iters = root_find(A_inds)
    logging.info(f'root finding: beta={beta}, iterations={iters}')
    S_thermal = thermal_entropy(beta)
    logging.info(f'calculate thermal entropy: S_thermal={S_thermal}')

    # test_GA_to_numpy_projection(A_inds=[0,1,4,5])
    # for beta in np.linspace(-1, 1, 10):
    #     test_thermal_entropy(beta, A_inds)
    # test_GA_to_numpy(A_inds)