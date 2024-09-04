import subprocess as sp
import shlex
import os
import csv
import logging
import argparse
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
# cd /home/yitan/Coding/deviation_ee/scripts/local
# python batch_job.py --L 12 --mode workflow
# cd /home/yitan/Coding/deviation_ee/data/obs_syk && cat expt_GA_L=12_report.csv

def check_syk_msc(L, seed, data_dir):
    data_dir = os.path.join(data_dir, "msc_syk")
    msc_file = os.path.join(data_dir, f"H_L={L}_seed={seed}.msc")
    if not os.path.isfile(msc_file):
        logging.info(f"msc_syk [{L}, {seed}] not found, start building ...")
        raise FileNotFoundError
    
def check_extrm_eigval(L, seed, data_dir):
    data_dir = os.path.join(data_dir, "extrm_eigval")
    evals_file = os.path.join(data_dir, f"eval_L={L}_seed={seed}.npy")
    if not os.path.isfile(evals_file):
        logging.info(f"extrm_eigval [{L}, {seed}] not found, start building ...")
        raise FileNotFoundError

def check_syk_vec(L, seed, tol, data_dir):
    data_dir = os.path.join(data_dir, "vec_syk_pm_z2_newtol")
    vec_file = os.path.join(data_dir, f"v_L={L}_seed={seed}_tol={tol}.vec")
    if not os.path.isfile(vec_file):
        logging.info(f"vec [{L}, {seed}, {tol}] not found, start building ...")
        raise FileNotFoundError
    
def check_obs(L, seed, tol, data_dir):
    data_dir = os.path.join(data_dir, "obs_syk")
    th_ent_file = os.path.join(data_dir, f"ent_entropy_L={L}_seed={seed}_tol={tol}.csv")
    expt_H_file = os.path.join(data_dir, f"expt_H_L={L}_seed={seed}_tol={tol}.csv")
    expt_H2_file = os.path.join(data_dir, f"expt_H2_L={L}_seed={seed}_tol={tol}.csv")
    if not os.path.isfile(th_ent_file) \
        or not os.path.isfile(expt_H_file) \
        or not os.path.isfile(expt_H2_file):
        logging.info(f"obs [{L}, {seed}, {tol}] not complete, start building ...")
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
    
def solve_extrm_eigval(L, seed, exec):
    args = shlex.split(exec + f" syk/solve_extrm_eigval.py --L {L} --seed {seed} --msc_dir data/msc_syk --res_dir data/extrm_eigval --gpu")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"solve_extrm_eigval failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception

def solve_syk_powermethod(L, seed, tol, exec):
    args = shlex.split(exec + f" syk/solve_syk_powermethod.py --L {L} --seed {seed} --tol {tol} --msc_dir data/msc_syk --vec_dir data/vec_syk_pm_z2_newtol --eval_dir data/extrm_eigval --gpu")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"solve_syk_powermethod failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception

def build_GA(L, seed, exec):
    LA = L // 2
    args = shlex.split(exec + f" syk/build_GA.py --L {L} --LA {LA} --seed {seed} --msc_dir data/msc_npy_GA --gpu 1 --n_groups 1")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"build_GA failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception
    
def build_G(L, seed, exec):
    args = shlex.split(exec + f" syk/build_G.py --L {L} --seed {seed} --msc_dir data/msc_syk --save True")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"build_G failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception
    
def measure_obs(L, seed, tol, exec):
    args = shlex.split(exec + f" syk/measure_obs.py --L {L} --seed {seed} --tol {tol} --msc_dir data/msc_syk --vec_dir data/vec_syk_pm_z2_newtol --obs_dir data/obs_syk --gpu --save True")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"measure_obs failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception

def measure_th(L, seed, tol, exec):
    LA = L // 2
    args = shlex.split(exec + f" syk/measure_th.py --L {L} --LA {LA} --seed {seed} --tol {tol} --op_dir data/msc_npy_GA --vec_dir data/vec_syk_pm_z2_newtol --obs_dir data/obs_syk --gpu --save True")
    res = sp.run(args, capture_output=True)
    if res.returncode != 0:
        logging.debug(f"measure_th failed\n \
                        {res.stdout.decode('utf-8')} \
                        {res.stderr.decode('utf-8')}")
        raise Exception

def gen_expt_GA_report(L, seeds, tols, data_dir):
    data_dir = os.path.join(data_dir, "obs_syk")
    expt_GA_data = np.zeros((len(tols), len(seeds)))
    for seed in tqdm(seeds, desc="gen_expt_GA_report"):
        iseed = seeds.index(seed)
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

    print(f"{'tol':<8}{'avg':<25}{'std':<25}{'max':<25}{'min':<25}")
    for itol, tol in enumerate(tols):
        print(f"{tol:<8}{expt_GA_avg[itol]:<25}{expt_GA_std[itol]:<25}{expt_GA_max[itol]:<25}{expt_GA_min[itol]:<25}")

    expt_GA_report = os.path.join(data_dir, f"expt_GA_L={L}_report.csv")
    with open(expt_GA_report, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['tol', 'avg', 'std', 'max', 'min'])
        for itol, tol in enumerate(tols):
            writer.writerow([tol, expt_GA_avg[itol], expt_GA_std[itol], expt_GA_max[itol], expt_GA_min[itol]])

def gen_expt_H_report(L, seeds, tols, data_dir):
    data_dir = os.path.join(data_dir, "obs_syk")
    expt_H_data = np.zeros((len(tols), len(seeds)))
    for seed in tqdm(seeds, desc="gen_expt_H_report"):
        iseed = seeds.index(seed)
        for itol, tol in enumerate(tols):
            expt_H_file = os.path.join(
                                data_dir,
                                f"expt_H_L={L}_seed={seed}_tol={tol}.csv")
            with open(expt_H_file, 'r') as f:
                rows = list(csv.reader(f))
                expt_H_data[itol, iseed] = float(rows[1][2])

    expt_H_avg = np.mean(expt_H_data, axis=1)
    expt_H_std = np.std(expt_H_data, axis=1)
    expt_H_max = np.max(expt_H_data, axis=1)
    expt_H_min = np.min(expt_H_data, axis=1)

    print(f"{'tol':<8}{'avg':<25}{'std':<25}{'max':<25}{'min':<25}")
    for itol, tol in enumerate(tols):
        print(f"{tol:<8}{expt_H_avg[itol]:<25}{expt_H_std[itol]:<25}{expt_H_max[itol]:<25}{expt_H_min[itol]:<25}")

    expt_H_report = os.path.join(data_dir, f"expt_H_L={L}_report.csv")
    with open(expt_H_report, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['tol', 'avg', 'std', 'max', 'min'])
        for itol, tol in enumerate(tols):
            writer.writerow([tol, expt_H_avg[itol], expt_H_std[itol], expt_H_max[itol], expt_H_min[itol]])

def gen_expt_H2_report(L, seeds, tols, data_dir):
    data_dir = os.path.join(data_dir, "obs_syk")
    expt_H2_data = np.zeros((len(tols), len(seeds)))
    for seed in tqdm(seeds, desc="gen_expt_H2_report"):
        iseed = seeds.index(seed)
        for itol, tol in enumerate(tols):
            expt_H2_file = os.path.join(
                                data_dir,
                                f"expt_H2_L={L}_seed={seed}_tol={tol}.csv")
            with open(expt_H2_file, 'r') as f:
                rows = list(csv.reader(f))
                expt_H2_data[itol, iseed] = float(rows[1][2])

    expt_H2_avg = np.mean(expt_H2_data, axis=1)
    expt_H2_std = np.std(expt_H2_data, axis=1)
    expt_H2_max = np.max(expt_H2_data, axis=1)
    expt_H2_min = np.min(expt_H2_data, axis=1)

    print(f"{'tol':<8}{'avg':<25}{'std':<25}{'max':<25}{'min':<25}")
    for itol, tol in enumerate(tols):
        print(f"{tol:<8}{expt_H2_avg[itol]:<25}{expt_H2_std[itol]:<25}{expt_H2_max[itol]:<25}{expt_H2_min[itol]:<25}")

    expt_H2_report = os.path.join(data_dir, f"expt_H2_L={L}_report.csv")
    with open(expt_H2_report, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['tol', 'avg', 'std', 'max', 'min'])
        for itol, tol in enumerate(tols):
            writer.writerow([tol, expt_H2_avg[itol], expt_H2_std[itol], expt_H2_max[itol], expt_H2_min[itol]])

def gen_tr_GA2_over_dim_report(L, seeds, data_dir):
    data_dir = os.path.join(data_dir, "obs_syk")
    tr_GA2_over_dim_data = np.zeros(len(seeds))
    for seed in tqdm(seeds, desc="gen_tr_GA2_over_dim_report"):
        iseed = seeds.index(seed)
        tr_GA2_over_dim_file = os.path.join(
                            data_dir,
                            f"tr_GA2_over_dim_L={L}_seed={seed}.csv")
        with open(tr_GA2_over_dim_file, 'r') as f:
            rows = list(csv.reader(f))
            tr_GA2_over_dim_data[iseed] = float(rows[1][2])

    tr_GA2_over_dim_avg = np.mean(tr_GA2_over_dim_data)
    tr_GA2_over_dim_std = np.std(tr_GA2_over_dim_data)
    tr_GA2_over_dim_max = np.max(tr_GA2_over_dim_data)
    tr_GA2_over_dim_min = np.min(tr_GA2_over_dim_data)

    print(f"{'avg':<25}{'std':<25}{'max':<25}{'min':<25}")
    print(f"{tr_GA2_over_dim_avg:<25}{tr_GA2_over_dim_std:<25}{tr_GA2_over_dim_max:<25}{tr_GA2_over_dim_min:<25}")

    tr_GA2_over_dim_report = os.path.join(data_dir, f"tr_GA2_over_dim_L={L}_report.csv")
    with open(tr_GA2_over_dim_report, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['avg', 'std', 'max', 'min'])
        writer.writerow([tr_GA2_over_dim_avg, tr_GA2_over_dim_std, tr_GA2_over_dim_max, tr_GA2_over_dim_min])

def wf_line_one(L, seeds, tols, exec, data_dir):
    for seed in tqdm(seeds, desc="wf_line_one"):
        try:
            check_syk_msc(L, seed, data_dir)
        except Exception as e:
            try:
                build_syk(L, seed, exec)
            except Exception as e:
                raise e
        
        try:
            check_extrm_eigval(L, seed, data_dir)
        except Exception as e:
            try:
                solve_extrm_eigval(L, seed, exec)
            except Exception as e:
                raise e

        try:
            for tol in tols:
                check_syk_vec(L, seed, tol, data_dir)
        except Exception as e:
            try:
                for tol in tols:
                    solve_syk_powermethod(L, seed, tol, exec)
            except Exception as e:
                raise e
            
        try:
            for tol in tols:
                check_obs(L, seed, tol, data_dir)
        except Exception as e:
            try:
                for tol in tols:
                    measure_obs(L, seed, tol, exec)
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
    except Exception:
        return
    
    try:
        wf_line_two(L, seeds, tols, exec, data_dir)
    except Exception:
        return

    gen_expt_H_report(L, seeds, tols, data_dir)
    gen_expt_H2_report(L, seeds, tols, data_dir)

    for seed in tqdm(seeds, desc="measure_th"):
        for tol in tols:
            try:
                measure_th(L, seed, tol, exec)
            except Exception:
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

    data_dir = os.path.join(data_dir_base, "extrm_eigval")
    for seed in tqdm(seeds, desc="Cleaning extrm_eigval"):
        for tol in tols:
            evals_file = os.path.join(data_dir, f"eval_L={L}_seed={seed}.npy")
            if os.path.isfile(evals_file):
                os.remove(evals_file)
    
    data_dir = os.path.join(data_dir_base, "obs_syk")
    for seed in tqdm(seeds, desc="Cleaning obs_syk"):
        for tol in tols:
            ent_entropy_file = os.path.join(data_dir, f"ent_entropy_L={L}_seed={seed}_tol={tol}.csv")
            expt_H_file = os.path.join(data_dir, f"expt_H_L={L}_seed={seed}_tol={tol}.csv")
            expt_H2_file = os.path.join(data_dir, f"expt_H2_L={L}_seed={seed}_tol={tol}.csv")
            th_ent_file = os.path.join(data_dir, f"thermal_entropy_L={L}_seed={seed}_tol={tol}.csv")
            expt_GA_file = os.path.join(data_dir, f"expt_GA_L={L}_seed={seed}_tol={tol}.csv")
            
            files = [
                ent_entropy_file,
                expt_H_file,
                expt_H2_file,
                th_ent_file,
                expt_GA_file
            ]

            for file in files:
                if os.path.isfile(file):
                    os.remove(file)
    
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
    seeds = [i for i in range(0, 20)]
    tols = [0.1, 0.01, 0.001]
    
    # if args.mode == 'workflow':
    #     workflow(L, seeds, tols, executable, data_dir)
    # elif args.mode == 'clean':
    #     clean(L, seeds, tols, data_dir)
    # elif args.mode == 'all':
    #     clean(L, seeds, tols, data_dir)
    #     workflow(L, seeds, tols, executable, data_dir)

    for seed in tqdm(seeds):
        build_GA(L, seed, executable)
        # build_G(L, seed, executable)
        # for tol in tols:
        #     measure_th(L, seed, tol, executable)

    # gen_tr_GA2_over_dim_report(L, seeds, data_dir)