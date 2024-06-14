#!/bin/bash -l
#SBATCH -J tst_sparse      # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -o %j.out   # Standard output and error log
#SBATCH -e %j.err
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem=12G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu_requeue,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

module load cuda/12.4.1-fasrc01
mamba activate qec_numerics
nvidia-smi
L=$1
LA=$2
seed=$3
python build_GA_cuda_sparse.py --L "$L" --LA "${LA}" --seed "${seed}" --save True 2>&1 | tee "build_GA_cuda_sparse_L=${L}_LA=${LA}_seed=${seed}.log"


