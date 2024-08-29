import os
import csv
import argparse
import logging
import numpy as np
from scipy.linalg import logm, expm
from scipy.optimize import root_scalar
from dynamite import config
from dynamite.states import State
from dynamite.computations import reduced_density_matrix

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

def load_GA(L, LA, seed) -> np.ndarray:
    """
    Load the subsystem Hamiltonian GA from file.
    And project it to the even parity subspace.
    Return:
        GA: shape (2**LA, 2**LA), the subsystem Hamiltonian
        This means GA is not projected to subspace
    """
    filename = f'GA_L={L}_LA={LA}_seed={seed}_dnm_decmp.npy'
    try:
        GA = np.load(os.path.join(args.op_dir, filename))
        logging.info(f'load GA: L={L}, LA={LA}, \
                     seed {seed} ...')
        return GA
    except:
        logging.error(f'GA file not found: L={L}, LA={LA}, \
                      seed {seed} ...')
        return None
    
def test_load_GA(L, LA, seed):
    GA = load_GA(L, LA, seed)
    assert GA.shape[0] == 2**LA, f'GA shape: {GA.shape}'

def load_state_vec(L, seed, tol):
    """
    Load the state vector from file.
    Return:
        v: dynamite.states.State, the state vector
        The reason for returning this formate is to use
        dynamite's native reduced density matrix method
    """
    filename_str = f'v_L={L}_seed={seed}_tol={tol}'
    try:
        v = State.from_file(os.path.join(args.vec_dir, filename_str))
        logging.info(f'load state vec: L={L}, seed {seed} ...')
        # return v.to_numpy()
        return v
    except:
        logging.error(f'state vec file not found: L={L}, seed {seed} ...')
        return None

def test_load_state_vec(L, seed, tol):
    v = load_state_vec(L, seed, tol)
    print(v.L)
    assert v.shape == (2**(L-1),), f'v shape: {v.shape}'

'''
Temporarily not using this method, because it's not fast.
def extend_v(v, L) -> np.ndarray:
    """
    Extend the loaded state vector into full Hilbert space.
    Return:
        v_ext: shape (2**L,), the extended state vector
    """
    v_ext = np.zeros(2**L, dtype=np.complex128)
    for i in range(2**L):
        i_binstr = np.binary_repr(i, width=L)
        # interleave even and odd basis
        if i_binstr.count('1') % 2 == 0:
            v_ext[i] = v[i // 2]
        else:
            v_ext[i] = 0
    return v_ext

def test_extend_v(v, L):
    v_ext = extend_v(v, L)
    assert v_ext.shape == (2**L,), f'v_ext shape: {v_ext.shape}'
'''

def extend_v(v, L) -> np.ndarray:
    """
    Extend the loaded state vector into full Hilbert space.
    Return:
        v_ext: shape (2**L,), the extended state vector
    """
    v_np = v.to_numpy()
    v_ext = np.zeros(2**L, dtype=np.complex128)
    for i in range(2**L):
        i_binstr = np.binary_repr(i, width=L)
        # interleave even and odd basis
        if i_binstr.count('1') % 2 == 0:
            v_ext[i] = v_np[i // 2]
        else:
            v_ext[i] = 0
    return v_ext

def test_extend_v(v, L):
    v_ext = extend_v(v, L)
    assert v_ext.shape == (2**L,), f'v_ext shape: {v_ext.shape}'

def get_v_rdm(v_ext, LA) -> np.ndarray:
    """
    Get reduced density matrix from the extended state vector.
    Need to use this method because it saves memory.
    If instead extending GA to GA\otimes I_Abar, it's equivalenet to doing
    2**L x 2**L full matrix. Will cause memory error.

    Parameters:
        # v: dynamite.states.State, the state vector
        v_ext: np.ndarray((2**L,), dtype=complex), the extended state vector

    Return:
        rdm: np.ndarray((2**LA, 2**LA), dtype=complex), 
            the reduced density matrix
    """
    logging.info(f'calculate reduced density matrix: LA={LA} ...')
    # rdm = reduced_density_matrix(v, np.arange(LA))
    L = int(np.log2(len(v_ext)))
    rdm = np.zeros((2**LA, 2**LA), dtype=np.complex128)
    for i in range(2**LA):
        for j in range(2**LA):
            for k in range(2**(L-LA)):
                # vary the first (L-LA) bits
                row_ind = k * 2**LA + i
                col_ind = k * 2**LA + j
                # Pictorially using |i >< j| to help understand,
                # |i><j| is not the actual matrix element
                rdm[i, j] += v_ext[row_ind] * np.conj(v_ext[col_ind])
    
    return rdm

def test_v_rdm(v_ext, LA):
    rdm = get_v_rdm(v_ext, LA)
    assert rdm.shape == (2**LA, 2**LA), f'rdm shape: {rdm.shape}'
    assert np.allclose(np.trace(rdm), 1), f'trace of rdm: {np.trace(rdm)}'

def expt_GA_rdm(GA, v_rdm, save=False, append=True):
    """
    Prepare the data for diff_GA_expt_thermal,
    so that in root finding, these calculations are not repeated.
    """
    logging.info(f'measure GA expectation value ...')
    expt = np.trace(GA @ v_rdm).real

    if save:
        csv_filepath = os.path.join(args.obs_dir, \
            f"expt_GA_L={args.L}_seed={args.seed}_tol={args.tol}.csv")
    
        if append:
            with open(csv_filepath, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([args.L, args.seed, expt])
        else:
            with open(csv_filepath, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','expt_GA'])
                writer.writerow([args.L, args.seed, expt])

    return expt

def thermal_energy(GA, beta):
    """
    Calculate the thermal energy of GA.
    """
    GA_dm = expm(-beta * GA).real
    Z = np.trace(GA_dm)
    logging.debug(f'Z: {Z}')
    thermal_energy = np.trace(GA_dm @ GA).real / Z
    logging.debug(f'thermal entropy: {thermal_energy}')
    return thermal_energy

def diff_GA_expt_thermal(expt_GA, GA, beta):
    """
    Diff two results.
    1. Measure the subsystem Hamiltonian GA using state vector |psi>.
    i.e., evaluating <psi| GA |psi> 
    2. Compute the thermal energy of GA,
    i.e., evaluating tr(GA * rho) where rho = exp(-beta * GA) / Z
    """
    GA_dm = expm(-beta * GA).real
    Z = np.trace(GA_dm)
    thermal_energy = np.trace(GA_dm @ GA).real / Z
    return expt_GA - thermal_energy

def root_find(expression, beta0=-0.1, beta1=0.1):
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
    
def thermal_entropy(GA, beta, save=False, append=True):
    """
    Calculate the thermal bound Delta S = S_vN(rho_GA(beta)), at the calculated beta.
    Here S_vN is the von-Neumann entropy.
    """
    GA_dm = expm(-beta * GA).real
    Z = np.trace(GA_dm)
    rho = GA_dm / Z
    S_thermal = -np.trace(rho @ logm(rho)).real

    if save:
        csv_filepath = os.path.join(args.obs_dir, \
            f"thermal_entropy_L={args.L}_seed={args.seed}_tol={args.tol}.csv")
    
        if append:
            with open(csv_filepath, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([args.L, args.seed, beta, S_thermal])
        else:
            with open(csv_filepath, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['L','seed','beta','S_thermal'])
                writer.writerow([args.L, args.seed, beta, S_thermal])

    return S_thermal

def pipeline(L, LA, seed, tol):
    GA = load_GA(L, LA, seed)
    v = load_state_vec(L, seed, tol)
    v_ext = extend_v(v, L)
    v_rdm = get_v_rdm(v_ext, LA)
    expt_GA = expt_GA_rdm(GA, v_rdm, save=args.save, append=False)
    logging.debug(f'expectation value of GA: {expt_GA}')
    expression = lambda beta: diff_GA_expt_thermal(expt_GA, GA, beta)
    beta, iters = root_find(expression)
    logging.info(f'root finding: beta={beta}, iterations={iters}')
    S_thermal = thermal_entropy(GA, beta, save=args.save, append=False)
    logging.info(f'calculate thermal entropy: S_thermal={S_thermal}')
    logging.info(f'maximal thermal entropy: LA*log2={LA*np.log(2)}')

def test():
    GA = load_GA(args.L, args.LA, args.seed)
    beta = -0.1
    logging.debug(f'thermal energy: {thermal_energy(GA, beta)}')
    test_load_GA(args.L, args.LA, args.seed)
    v = load_state_vec(args.L, args.seed, args.tol)
    v_ext = extend_v(v, args.L)
    test_v_rdm(v_ext, args.LA)
    

if __name__ == '__main__':
    # test()
    pipeline(args.L, args.LA, args.seed, args.tol)