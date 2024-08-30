import subprocess as sp
import multiprocessing as mp
import shlex
import os
import csv
import logging
import argparse
from typing import Literal
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, required=True, help='system size')
parser.add_argument('--mode',
                    type=str,
                    required=False, 
                    default='workflow',
                    choices=['workflow', 'clean', 'all'],
                    help='task mode')
args = parser.parse_args()

# Usage:
# python batch_job.py --L 12 --mode workflow
# cd /home/yitan/Coding/deviation_ee/data/obs_syk | cat expt_GA_L=12_report.csv

def check_syk_msc(L, seed, data_dir):
    data_dir = os.path.join(data_dir, "msc_syk")
    msc_file = os.path.join(data_dir, f"H_L={L}_seed={seed}.msc")
    if not os.path.isfile(msc_file):
        logging.info(f"msc_syk [{L}, {seed}] not found, start building ...")
        raise FileNotFoundError

def check_syk_vec(L, seed, tol, data_dir):
    data_dir = os.path.join(data_dir, "vec_syk_pm_z2_newtol")
    vec_file = os.path.join(data_dir, f"v_L={L}_seed={seed}_tol={tol}.vec")
    if not os.path.isfile(vec_file):
        logging.info(f"vec [{L}, {seed}, {tol}] not found, start building ...")
        raise FileNotFoundError

def check_GA(L, seed, data_dir):
    LA = L // 2
    data_dir = os.path.join(data_dir, "msc_npy_GA")
    GA_file = os.path.join(data_dir, f"GA_L={L}_LA={LA}_seed={seed}_dnm_decmp.npy")
    if not os.path.exists(GA_file):
        logging.info(f"GA_npy [{L}, {seed}] not found, start building ...")
        raise FileNotFoundError

def build_syk(L, seed, exec):
    # need to grant o+w permission to all data/* subdirectory
    args = shlex.split(exec + f" syk/build_syk.py --L {L} --seed {seed} --msc_dir data/msc_syk --gpu 1")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"build_syk failed\n \
                      {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception

def solve_syk_powermethod(L, seed, tol, exec):
    args = shlex.split(exec + f" syk/solve_syk_powermethod.py --L {L} --seed {seed} --tol {tol} --msc_dir data/msc_syk --vec_dir data/vec_syk_pm_z2_newtol --gpu")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"solve_syk_powermethod failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception

def build_GA(L, seed, exec):
    LA = L // 2
    args = shlex.split(exec + f" syk/build_GA.py --L {L} --LA {LA} --seed {seed} --msc_dir data/msc_npy_GA --gpu 1")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"build_GA failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception

def measure_th(L, seed, tol, exec):
    LA = L // 2
    args = shlex.split(exec + f" syk/measure_thermal_entropy.py --L {L} --LA {LA} --seed {seed} --tol {tol} --op_dir data/msc_npy_GA --vec_dir data/vec_syk_pm_z2_newtol --obs_dir data/obs_syk --gpu --save")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"measure_th failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception

def gen_expt_GA_report(L, seeds, tols, data_dir):
    data_dir = os.path.join(data_dir, "obs_syk")
    expt_GA_data = np.zeros((len(tols), len(seeds)))
    for iseed, seed in tqdm(enumerate(seeds), desc="gen_expt_GA_report"):
        for itol, tol in tqdm(enumerate(tols), desc="gen_expt_GA_report_tol"):
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
    for seed in tqdm(seeds, desc="wf_line_one"):
        try:
            logging.debug(f"Checking syk_msc [{L}, {seed}]")
            check_syk_msc(L, seed, data_dir)
        except Exception as e:
            try:
                build_syk(L, seed, exec)
            except Exception as e:
                raise e
        
        try:
            for tol in tols:
                check_syk_vec(L, seed, tol, data_dir)
        except Exception as e:
            try:
                solve_syk_powermethod(L, seed, tol, exec)
            except Exception as e:
                raise e

def wf_line_two(L, seeds, tols, exec, data_dir):
    for seed in tqdm(seeds, desc="wf_line_two"):
        try:
            check_GA(L, seed, data_dir)
        except Exception as e:
            for tol in tols:
                try:
                    build_GA(L, seed, exec)
                except Exception as e:
                    raise e
    return True

def workflow(L, seeds, tols, exec, data_dir):
    try:
        wf_line_one(L, seeds, tols, exec, data_dir)
    except Exception as e:
        return
    try:
        wf_line_two(L, seeds, tols, exec, data_dir)
    except Exception as e:
        return
    gen_expt_GA_report(L, seeds, tols, data_dir)

def clean(L, seeds, tols, data_dir_base):
    data_dir = os.path.join(data_dir_base, "msc_syk")
    for seed in tqdm(seeds, desc="Cleaning msc_syk"):
        msc_file = os.path.join(data_dir, f"H_L={L}_seed={seed}.msc")
        if os.path.isfile(msc_file):
            os.remove(msc_file)
    
    data_dir = os.path.join(data_dir_base, "vec_syk_pm_z2_newtol")
    for seed in tqdm(seeds, desc="Cleaning vec_syk_pm_z2_newtol"):
        for tol in tols:
            vec_file = os.path.join(data_dir, f"v_L={L}_seed={seed}_tol={tol}.vec")
            metadata_file = os.path.join(data_dir, f"v_L={L}_seed={seed}_tol={tol}.metadata")
            if os.path.isfile(vec_file):
                os.remove(vec_file)
            if os.path.isfile(metadata_file):
                os.remove(metadata_file)
    
    data_dir = os.path.join(data_dir_base, "msc_npy_GA")
    for seed in tqdm(seeds, desc="Cleaning msc_npy_GA"):
        LA = L // 2
        GA_file = os.path.join(data_dir, f"GA_L={L}_LA={LA}_seed={seed}_dnm_decmp.npy")
        if os.path.isfile(GA_file):
            os.remove(GA_file)
    
    data_dir = os.path.join(data_dir_base, "obs_syk")
    for seed in tqdm(seeds, desc="Cleaning obs_syk"):
        for tol in tols:
            expt_GA_file = os.path.join(data_dir, f"expt_GA_L={L}_seed={seed}_tol={tol}.csv")
            if os.path.isfile(expt_GA_file):
                os.remove(expt_GA_file)
    
    expt_GA_report = os.path.join(data_dir, f"expt_GA_L={L}_report.csv")
    if os.path.isfile(expt_GA_report):
        os.remove(expt_GA_report)
    
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
    
    if args.mode == 'workflow':
        workflow(L, seeds, tols, executable, data_dir)
    elif args.mode == 'clean':
        clean(L, seeds, tols, data_dir)
    elif args.mode == 'all':
        clean(L, seeds, tols, data_dir)
        workflow(L, seeds, tols, executable, data_dir)