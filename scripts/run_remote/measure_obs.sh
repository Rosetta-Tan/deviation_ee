#!/bin/bash -l
#SBATCH -J measure_obs      # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -o %j.out        # File to which STDOUT will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -e %j.err        # File to which STDERR will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem=4G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1

L=12
seed_start=0
seed_end=1
tol=0.001
vec_dir="/n/holyscratch01/yao_lab/ytan/deviation_ee/vec_syk_pm_z2_newtol"
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/measure_obs.py --L "$L" --seeds "${seed_start}"-"${seed_end}" --tol "${tol}" --vec_dir "${vec_dir}" --gpu --save True