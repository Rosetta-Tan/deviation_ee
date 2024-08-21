#!/bin/bash -l
#SBATCH -J solve_powermethod
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=20G
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1

L=12
seed=0
vec_dir="/n/holyscratch01/yao_lab/ytan/deviation_ee/vec_syk_pm_z2_newtol"
dir="/n/holyscratch01/yao_lab/ytan/deviation_ee/output/20240425_powermethod_z2_newtol"
tol=0.001
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/solve_syk_powermethod.py --vec_dir "${vec_dir}" --log_dir "${dir}" -L "${L}" --seed "${seed}" --tol "${tol}" --gpu