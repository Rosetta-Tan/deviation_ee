import subprocess as sp
import multiprocessing as mp
import shlex
import os
import csv
import logging
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, required=True, help='system size')
args = parser.parse_args()

def check_syk_msc(L, seed, data_dir):
    data_dir = os.path.join(data_dir, "msc_syk")
    msc_file = os.path.join(data_dir, f"H_L={L}_seed={seed}.msc")
    if not os.path.isfile(msc_file):
        logging.debug(f"{msc_file} not found")
        return False
    return True

def check_syk_vec(L, seed, tol, data_dir):
    data_dir = os.path.join(data_dir, "vec_syk_pm_z2_newtol")
    vec_file = os.path.join(data_dir, f"v_L={L}_seed={seed}_tol={tol}.vec")
    if not os.path.isfile(vec_file):
        logging.debug(f"{vec_file} not found")
        return False
    return True

def check_GA(L, seed, data_dir):
    LA = L // 2
    data_dir = os.path.join(data_dir, "msc_npy_GA")
    GA_file = os.path.join(data_dir, f"GA_L={L}_LA={LA}_seed={seed}_dnm_decmp.npy")
    if not os.path.exists(GA_file):
        logging.debug(f"{GA_file} not found")
        return False
    return True

def build_syk(L, seed, exec):
    args = shlex.split(exec + f" syk/build_syk.py --L {L} --seed {seed} --dirc ./ --gpu 1")
    res = sp.run(args, capture_output=True)
    return res.returncode

def solve_syk_powermethod(L, seed, tol):
    args = shlex.split(exec + f" syk/solve_syk_powermethod.py --L {L} --seed {seed} --tol {tol}")
    res = sp.run(args, capture_output=True)
    return res.returncode

def build_GA(L, seed, exec):
    args = shlex.split(exec + f" syk/build_GA.py --L {L} --seed {seed}")
    res = sp.run(args, capture_output=True)
    return res.returncode

def measure_th(L, seed, tol, exec):
    args = shlex.split(exec + f" syk/measure_thermal_entropy.py --L {L} --seed {seed} --tol {tol}")
    res = sp.run(args, capture_output=True)
    return res.returncode

def gen_expt_GA_report(L, seeds, tols, data_dir):
    data_dir = os.path.join(data_dir, "obs_syk")
    expt_GA_data = np.zeros((len(tols), len(seeds)))
    for iseed, seed in enumerate(seeds):
        for itol, tol in enumerate(tols):
            expt_GA_file = os.path.join(
                                data_dir,
                                f"expt_GA_L={L}_seed={seed}_tol={tol}.csv")    
            with open(expt_GA_file, 'r') as f:
                rows = list(csv.reader(f))
                expt_GA_data[itol, iseed] = float(rows[1][2])

    
    expt_GA_avg = np.mean(expt_GA_data, axis=1)
    expt_GA_std = np.std(expt_GA_data, axis=1)
    expt_GA_max = np.max(expt_GA_data, axis=1)
    expt_GA_min = np.min(expt_GA_data, axis=1)
    
    with open(os.path.join(data_dir, f"expt_GA_L={L}_report.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["tol", "expt_GA_avg", "expt_GA_std", "expt_GA_max", "expt_GA_min"])
        for itol, tol in enumerate(tols):
            writer.writerow([tol,
                             expt_GA_avg[itol],
                             expt_GA_std[itol],
                             expt_GA_max[itol],
                             expt_GA_min[itol]])
    
def wf_line_one(L, seeds, tols, exec, data_dir):
    for seed in seeds:
        try:
            check_syk_msc(L, seed, data_dir)
        except Exception as e:
            logging.debug(e)
            build_syk_res = build_syk(L, seed, exec)
            if build_syk_res != 0:
                logging.debug(f"build_syk failed with code {build_syk_res}")
                return False
        
        try:
            for tol in tols:
                check_syk_vec(L, seed, tol, data_dir)
        except Exception as e:
            logging.debug(e)
            for tol in tols:
                solve_vec_res = solve_syk_powermethod(L, seed, tol)
                if solve_vec_res != 0:
                    logging.debug(f"solve_syk_powermethod failed with code {solve_vec_res}")
                    return False
            
        return True

def wf_line_two(L, seeds, tols, exec, data_dir):
    for seed in seeds:
        try:
            check_GA(L, seed, data_dir)
        except Exception as e:
            logging.debug(e)
            for tol in tols:
                measure_th_res = measure_th(L, seed, tol, exec)
                if measure_th_res != 0:
                    logging.debug(f"measure_th failed with code {measure_th_res}")
                    return False
    return True

def workflow(L, seeds, tols, exec, data_dir):
    if not wf_line_one(L, seeds, tols, exec, data_dir):
        return False
    if not wf_line_two(L, seeds, tols, exec, data_dir):
        return False
    gen_expt_GA_report(L, seeds, tols, data_dir)
    
if __name__ == '__main__':
    from config import config
    os.environ['TZ'] = 'EST5EDT'
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    machine = "hqi401b"
    executable = config[machine]["exec"]
    data_dir = config[machine]["data_dir"]
    L = args.L
    seeds = [i for i in range(20)]
    tols = [0.1, 0.01, 0.001]
    
    workflow(L, seeds, tols, executable, data_dir)