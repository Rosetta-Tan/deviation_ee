#!/bin/bash -l
#SBATCH -J GA_dnm_decmp      # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events
#SBATCH -o %j.out        # File to which STDOUT will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -e %j.err        # File to which STDERR will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -t 0-12:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 20                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem=120G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100'

# usage: sbatch run_dynamite_decmp_itgrt.sh L LA n_groups

L=$1
LA=$2
n_groups=$3

# singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python dynamite_decmp_itgrt.py --L "$L" --LA "${LA}" --seed "${seed}" --gpu 1 --n_groups "${n_groups}"

echo "Running script with parameters: L=${L}, LA=${LA}, n_groups=${n_groups}"

for seed in {1..19}
do
    srun --exclusive -n 1 singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python dynamite_decmp_itgrt.py --L "$L" --LA "${LA}" --seed "${seed}" --gpu 1 --n_groups "${n_groups}" &
done

wait