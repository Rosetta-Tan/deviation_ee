#!/bin/bash -l
#SBATCH -J measure_th
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /n/home01/ytan/scratch/deviation_ee/output/workflow/%j_mt.out
#SBATCH -e /n/home01/ytan/scratch/deviation_ee/output/workflow/%j_mt.err
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=15G
#SBATCH -p gpu,gpu_requeue
#SBATCH --gres=gpu:1

L=$1
seed=$2
tol=$3
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/measure_thermal_entropy.py --L "$L" --seed "${seed}" --tol "${tol}" --gpu --save