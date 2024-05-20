#!/bin/bash -l
###### for CPU #####
# #SBATCH -J solve_syk_powermethod        # Job name
# #SBATCH --account=yao_lab
# #SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH -o /n/home01/ytan/deviation_ee/tests/step_time_test/%A_%a.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
# #SBATCH -e /n/home01/ytan/deviation_ee/tests/step_time_test/%A_%a.err  # File to which STDERR will be written, %j inserts jobid
# #SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
# #SBATCH -n 64                # Number of tasks
# #SBATCH -N 4                # Ensure that all cores are on one machine
# #SBATCH --ntasks-per-node=16
# #SBATCH --mem-per-cpu=8G       # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH -p shared	    # Partition to submit to

# module load python
# module load gcc openmpi
# mamba activate /n/holystore01/LABS/yao_lab/Lab/mamba_envs/complex-opt-0

# srun --mpi=pmix -n $SLURM_NTASKS python -O /n/home01/ytan/deviation_ee/syk/solve_syk_powermethod.py -L 14 --seed $SLURM_ARRAY_TASK_ID --tol_coeff 0.01 --vec_dir /n/home01/ytan/deviation_ee/tests/step_time_test/

##### for GPU #####
#SBATCH -J solve_syk_powermethod      # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -o /n/home01/ytan/deviation_ee/tests/step_time_test/%A_%a.out  # File to which STDOUT will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -e /n/home01/ytan/deviation_ee/tests/step_time_test/%A_%a.err  # File to which STDERR will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem=32G       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu_requeue,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

mamba activate qec_numerics
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/solve_syk_powermethod.py -L 14 --seed $SLURM_ARRAY_TASK_ID --tol_coeff 0.01 --vec_dir /n/home01/ytan/deviation_ee/tests/step_time_test/


##### post processing #####
# catch the expression "time" and "error" in the output file
