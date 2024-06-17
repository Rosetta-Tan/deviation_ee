import os, sys, csv, argparse
from dynamite import config
from dynamite.states import State
from dynamite.operators import zero, op_sum, op_product, identity, Operator
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
parser.add_argument('--LA', required=True, type=int, help='subsystem size')
parser.add_argument('--seed', required=True, type=int, help='random seed')
parser.add_argument('--tol', required=True, type=float, \
                    help='tolerance for powermethod solving state vector')
parser.add_argument('--log_dir', type=str, required=False, \
                    default='/n/home01/ytan/scratch/deviation_ee/output/2024602_thermal_bound', \
                        help='directory to save log files')
parser.add_argument('--op_dir', type=str, required=False, \
                    default='/n/home01/ytan/scratch/deviation_ee/msc_npy_GA', \
                        help='directory to save state vectors')
parser.add_argument('--vec_dir', type=str, required=False, \
                    default='/n/home01/ytan/scratch/deviation_ee/vec_syk_pm_z2_newtol', \
                        help='directory to save state vectors')
parser.add_argument('--obs_dir', type=str, required=False, \
                    default='/n/home01/ytan/scratch/deviation_ee/obs_syk', \
                        help='directory to save obs files')
parser.add_argument('--gpu', required=False, action='store_true', help='use GPU')
parser.add_argument('--save', type=bool, default=False, help='save data')
args = parser.parse_args()

if args.gpu:
    config.initialize(gpu=True)
config.shell=True  # not using shift-and-invert, so can use shell matrix

def save_data(csv_filepath, *data, append=True):
    with open(csv_filepath, 'a' if append else 'w') as file:
        writer = csv.writer(file)
        writer.writerow([args.L, args.seed, *data])

def load_GA() -> np.ndarray:
    """
    Load the subsystem Hamiltonian GA from file.
    And project it to the even parity subspace.
    # TODO: check if the GA is in the even parity subspace
    """
    filename = f'GA_L={args.L}_LA={args.LA}_seed={args.seed}_dnm_decmp.npy'
    try:
        GA = np.load(os.path.join(args.op_dir, filename))
        logging.info(f'load GA: L={args.L}, LA={args.LA}, \
                     seed {args.seed} ...')
        return GA
    except:
        logging.error(f'GA file not found: L={args.L}, LA={args.LA}, \
                      seed {args.seed} ...')
        return None
    
def test_load_GA():
    GA = load_GA()
    assert GA.shape[0] == 2**(args.LA), f'GA shape: {GA.shape}'

def extend_GA_IAbar(GA: np.ndarray, L, LA):
    """
    GA \otimes I_\bar{A}, and project to parity-even subspace
    """
    LAbar = L - LA
    GA_IAbar = np.kron(GA, np.eye(2**LAbar, 2**LAbar))
    projector = np.zeros((2**(L-1), 2**L), dtype=int)
    for i in range(2**L):
        basis_vec = np.binary_repr(i, width=L)
        if basis_vec.count('1') % 2 == 0:
            projector[i//2, i] = 1
    GA_IAbar = projector @ GA_IAbar @ projector.T
    return GA_IAbar

def test_extend_GA_IAbar(L, LA):
    """
    - Test the conversion from GA to numpy array (dimensionality issue)
    - Test the trace of GA^2/2^L_A, 
        to see if it matches the theoretical result
    """
    GA = load_GA()
    GA_IAbar = extend_GA_IAbar(GA, L, LA)
    print('(debug) trace of (GA\otimes IAbar)^2/2^(L-1): ', \
        np.trace(GA_IAbar @ GA_IAbar) / 2**(L-1))
    assert GA_IAbar.shape == (2**(L-1), 2**(L-1)), \
        f'GA_IAbar shape: {GA_IAbar.shape}'

def test_GA_dm(beta):
    A_inds = list(range(0, config.L//2))
    GA = load_GA(A_inds)
    GA_numpy = GA_to_numpy(GA, A_inds)
    GA_dm = expm(-beta * GA_numpy)
    assert GA_dm.shape == (2**len(A_inds), 2**len(A_inds)), f'GA_dm shape: {GA_dm.shape}'

def test_thermal_energy(beta):
    # TODO
    pass

def test_thermal_entropy(beta, A_inds):
    GA = load_GA(A_inds)
    GA_numpy = GA_to_numpy(GA, A_inds)
    GA_dm = expm(-beta * GA_numpy).toarray()
    print('rho shape: ', rho.shape)
    Z = np.trace(GA_dm)
    rho = GA_dm / Z
    S_thermal = -np.trace(rho @ logm(rho)).real
    print(f'(debug): beta={beta}, S_thermal={S_thermal}')
    return S_thermal


def load_state_vec(L, seed, tol) -> np.ndarray:
    filename_str = f'v_L={L}_seed={seed}_tol={tol}'
    try:
        v = State.from_file(os.path.join(args.vec_dir, filename_str))
        logging.info(f'load state vec: L={L}, seed {seed} ...')
        return v.to_numpy()
    except:
        logging.error(f'state vec file not found: L={L}, seed {seed} ...')
        return None

def test_load_state_vec(L, seed, tol):
    v = load_state_vec(L, seed, tol)
    assert v.shape == (2**(L-1),), f'v shape: {v.shape}'

def expt_GA_IAbar(v, GA_IAbar, L, LA):
    """
    Prepare the data for diff_GA_expt_thermal,
    so that in root finding, these calculations are not repeated.
    """
    expt_GA = (v @ GA_IAbar @ v).real
    logging.info(f'measure GA expectation value: L={L}, seed {args.seed} ...')
    
    return expt_GA

def diff_GA_expt_thermal(expt_GA, GA, beta):
    """
    Diff two results.
    1. Measure the subsystem Hamiltonian GA using state vector |psi>.
    i.e., evaluating <psi| GA |psi> 
    2. Compute the thermal energy of GA,
    i.e., evaluating tr(GA * rho) where rho = exp(-beta * GA) / Z
    """
    GA_dm = expm(-beta * GA).real
    Z = np.trace(GA_dm.trace)
    thermal_energy = np.trace(GA_dm @ GA) / Z
    return expt_GA - thermal_energy

def root_find(expression, beta0=-100, beta1=100):
    """
    Find the beta that gives the correct thermal energy.
    Using bisect method.
    i.e., solving tr(GA * rho(beta)) = <psi| GA |psi> for beta
    
    Parameters:
    - expression: a lambda function of beta
    """
    print('at beta0: ', expression(beta0))
    print('at beta1: ', expression(beta1))
    sol = root_scalar(expression, bracket=[beta0, beta1])
    return sol.root, sol.iterations
    
def thermal_entropy(GA, beta, save=False):
    """
    Calculate the thermal bound Delta S = S_vN(rho_GA(beta)), at the calculated beta.
    Here S_vN is the von-Neumann entropy.
    """
    GA_dm = expm(-beta * GA).toarray()
    Z = np.trace(GA_dm)
    rho = GA_dm / Z
    S_thermal = -np.trace(rho @ logm(rho)).real

    if save:
        csv_filepath = os.path.join(args.obs_dir, f"thermal_entropy_L={args.L}_seed={args.seed}_tol={args.tol}.csv")
        if not os.path.exists(csv_filepath):
            with open(csv_filepath, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','beta','S_thermal'])
        save_data(csv_filepath, beta, S_thermal, append=True)
    return S_thermal

def pipeline(L, LA, seed, tol):
    GA = load_GA()
    print('here')
    GA_IAbar = extend_GA_IAbar(GA, L, LA)
    v = load_state_vec(L, seed, tol)
    expt_GA = expt_GA_IAbar(v, GA_IAbar, L, LA)
    expression = lambda beta: diff_GA_expt_thermal(expt_GA, GA, beta)
    beta, iters = root_find(expression)
    logging.info(f'root finding: beta={beta}, iterations={iters}')
    S_thermal = thermal_entropy(beta, save=args.save)
    logging.info(f'calculate thermal entropy: S_thermal={S_thermal}')

def test():
    test_load_state_vec(args.L, args.seed, args.tol)

if __name__ == '__main__':
    test()