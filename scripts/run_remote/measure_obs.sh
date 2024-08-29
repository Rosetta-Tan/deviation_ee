#!/bin/bash -l
#SBATCH -J measure_obs
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /n/home01/ytan/scratch/deviation_ee/output/workflow/%j_mo.out
#SBATCH -e /n/home01/ytan/scratch/deviation_ee/output/workflow/%j_mo.err
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=15G
#SBATCH -p gpu,gpu_requeue
#SBATCH --gres=gpu:1

L=$1
seed_start=$2
seed_end=$((seed_start+1))
tol=$3
vec_dir="/n/holyscratch01/yao_lab/ytan/deviation_ee/vec_syk_pm_z2_newtol"
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/measure_obs.py --L "$L" --seeds "${seed_start}"-"${seed_end}" --tol "${tol}" --vec_dir "${vec_dir}" --gpu --save True