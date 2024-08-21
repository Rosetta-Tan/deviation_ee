#!/bin/bash -l 
#SBATCH -J build_syk
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

L=$1
seed=$2
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/build_syk.py -L "$L" --seed "${seed}" --gpu 1