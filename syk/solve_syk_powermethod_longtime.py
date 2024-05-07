import os, argparse, csv, subprocess
from dynamite import config
from dynamite.states import State
from dynamite.operators import op_sum, op_product, identity, Operator
from dynamite.extras import majorana
from dynamite.subspaces import Parity
from itertools import combinations
import numpy as np
from scipy.special import comb
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from timeit import default_timer
from datetime import timedelta
import sys
import logging

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[FlushStreamHandler()])

parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, required=True, help='system size')
parser.add_argument('-J', type=float, required=False, default=1.0, help='coupling strength')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--msc_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk',  help='msc file directory')
parser.add_argument('--log_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/output',  help='log file directory')
parser.add_argument('--vec_dir', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/vec_syk/powermethod',  help='vector file directory')   # for saving the evolved state
parser.add_argument('--tol', type=float, required=False, default=1e-9, help='tolerance coefficient of energy variance of states for power method')
parser.add_argument('--gpu', required=False, action='store_true', help='use GPU')
parser.add_argument('--duration', type=int, required=False, default=86400, help='duration of the computation in seconds (default: 1 day)')
parser.add_argument('--resume', type=int, required=False, default=0, help='resume from the last saved state')  # 0: False, 1: True
args = parser.parse_args()

# for GPU
if args.gpu:
    config.initialize(gpu=True)
config.shell=True  # not using shift-and-invert, so can use shell matrix
config.subspace = Parity('even')
config.L = args.L
N = 2*args.L # number of Majoranas
if not os.path.isdir(args.msc_dir):
    os.mkdir(args.msc_dir)
if not os.path.isdir(args.vec_dir):
    os.mkdir(args.vec_dir)

def get_ckpt_err_filename():
    command = f'find {args.log_dir} -regex ".*syk_pm_long_L={args.L}_seed={args.seed}_tol={args.tol}.*ckpt.*\.err" | sort | tail -n 2 | head -n 1'
    filename = "/n" + subprocess.check_output(command, shell=True).decode('utf-8').split('\n')[0][2:]
    return filename

def recover_timedelta():
    filename = get_ckpt_err_filename()
    with open(filename, 'r') as f:
        lines = f.readlines()
        line=lines[-1]
        # locate where the substring 'time elapsed' starts
        start = line.find('time elapsed')
        # extract the substring starting from 'time elapsed'
        timedelta_str = line[start+len('time elapsed')+2:]
    
    if 'days' in timedelta_str or 'day' in timedelta_str:
        days_str, time_components_str = timedelta_str.split(', ')
    else:
        days_str = '0 days'
        time_components_str = timedelta_str
    days = int(days_str.split()[0])  # Extract days as an integer
    time_components = time_components_str.split(':')
    hours = int(time_components[0])
    minutes = int(time_components[1])
    seconds, microseconds = map(float, time_components[2].split('.'))
    delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

def recover_pm_step():
    filename = get_ckpt_err_filename()
    with open(filename, 'r') as f:
        lines = f.readlines()
        line=lines[-1]
        # locate where the substring 'pm_steps' starts
        start = line.find('pm_step') + len('pm_step') + 2
        # read the substring starting from 'pm_steps', ending before the next ',' character
        pm_step_str = line[start:line.find(',', start)]
        pm_step = int(pm_step_str)
    return pm_step

def get_extrm_eigval():
    with open('/n/home01/ytan/deviation_ee/syk/extrm_eigvals.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for i in range(1, len(rows)):
            if int(rows[i][0]) == args.L and int(rows[i][1]) == args.seed:
                return float(rows[i][2])
    return None

LAMBDA = abs(get_extrm_eigval())+0.001  # the tolerance was set to ~1e-5, so we add 0.001 to the largest eigenvalue

def cal_energy_variance(H, v):
    Hv = H.dot(v)
    return Hv.dot(Hv).real

def load_H():
    if not os.path.exists(os.path.join(args.msc_dir,f'H_L={args.L}_seed={args.seed}.msc')):
        logging.error(f'H_L={args.L}_seed={args.seed}.msc does not exist.')
    H = Operator.load(os.path.join(args.msc_dir,f'H_L={args.L}_seed={args.seed}.msc'))
    return H

def diff_state_vecs(v0, v1):
    v_diff = v1.copy()
    v_diff.axpy(-1.0, v0)
    return v_diff

def power_method_like_evol(H, op1, op2, v0, start, init_pm_steps):  # start: start time of the whole script
    # initialize
    converged = False
    v0.normalize()
    num_pm_steps = init_pm_steps
    logging.debug(f'num_pm_steps: {num_pm_steps}, energy variance: {cal_energy_variance(H, v0)}')
    v1 = op1.dot(v0)
    v2 = op2.dot(v1)
    v2.normalize()
    num_pm_steps += 1
    logging.debug(f'num_pm_steps: {num_pm_steps}, energy variance: {cal_energy_variance(H, v2)}')
    while default_timer()-start < args.duration and not converged:
        v0, v2 = v2, v0  # swap the two states
        v1 = op1.dot(v0)
        v2 = op2.dot(v1)
        v2.normalize()
        num_pm_steps += 1
        if num_pm_steps % 100 == 0:
            logging.debug(f'num_pm_steps: {num_pm_steps}, energy variance: {cal_energy_variance(H, v2)}')
        if cal_energy_variance(H, v2) < args.tol:
            converged = True
            break
    return v2, num_pm_steps, converged

def main():
    start = default_timer()
    if args.resume:
        time_shift = recover_timedelta()
    else:
        time_shift = timedelta(seconds=0)

    H = (1/4) * load_H()
    logging.info(f'load Hamiltonian, L={args.L}, seed={args.seed}, to be solved at tol {args.tol}; time elapsed: {timedelta(seconds=default_timer()-start) + time_shift}')
    op1 = (LAMBDA/4)*identity() - H
    op2 = (LAMBDA/4)*identity() + H
    logging.info(f'construct the two operators involved in power method, L={args.L}, seed={args.seed}; time elapsed: {timedelta(seconds=default_timer()-start) + time_shift}')
    
    if args.resume == 1:
        v0 = State.from_file(os.path.join(args.vec_dir,f'v_L={args.L}_seed={args.seed}_tol={args.tol}_ckpt'))
        init_pm_steps = recover_pm_step()
        logging.info(f'resume from the last saved state, L={args.L}, seed={args.seed}, tol={args.tol}; time elapsed: {timedelta(seconds=default_timer()-start) + time_shift}')
    else:
        # check whether v0 has been saved
        if os.path.exists(os.path.join(args.vec_dir,f'v0_L={args.L}_seed={args.seed}_tol={args.tol}.metadata')) and \
            os.path.exists(os.path.join(args.vec_dir,f'v0_L={args.L}_seed={args.seed}_tol={args.tol}.vec')):
            v0 = State.from_file(os.path.join(args.vec_dir,f'v0_L={args.L}_seed={args.seed}_tol={args.tol}'))
            logging.info(f'load existing v0, L={args.L}, seed={args.seed}, tol={args.tol}; time elapsed: {timedelta(seconds=default_timer()-start)}')
        else:
            v0 = State(state='random', seed=0)
            v0.save(os.path.join(args.vec_dir,f'v0_L={args.L}_seed={args.seed}_tol={args.tol}'))
            logging.info(f'generate v0, L={args.L}, seed={args.seed}, tol={args.tol}; time elapsed: {timedelta(seconds=default_timer()-start) + time_shift}')

    # perform computation
    if args.resume:
        v, num_pm_steps, converged = power_method_like_evol(H, op1, op2, v0, start=start, init_pm_steps=init_pm_steps)
    else:
        v, num_pm_steps, converged = power_method_like_evol(H, op1, op2, v0, start=start, init_pm_steps=0)
    logging.info(f'evolve state with power method, L={args.L}, seed={args.seed}; pm_step: {num_pm_steps}, time elapsed: {timedelta(seconds=default_timer()-start) + time_shift}')

    # save data
    if converged:
        v.save(os.path.join(args.vec_dir,f'v_L={args.L}_seed={args.seed}_tol={args.tol}'))
        print(f'L={args.L}, seed={args.seed}, tol={args.tol}, pm_step={num_pm_steps}', flush=True)
    else:
        v.save(os.path.join(args.vec_dir,f'v_L={args.L}_seed={args.seed}_tol={args.tol}_ckpt'))

if __name__ == '__main__':
    main()
