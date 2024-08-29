#!/bin/bash -l 
#SBATCH -J build_syk
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /n/home01/ytan/scratch/deviation_ee/output/workflow/%j_bs.out
#SBATCH -e /n/home01/ytan/scratch/deviation_ee/output/workflow/%j_bs.err
#SBATCH -t 2-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=20G
#SBATCH -p yao_gpu
#SBATCH --gres=gpu:1

L=$1
seed=$2
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/build_syk.py -L "$L" --seed "${seed}" --gpu 1