#!/bin/bash -l
#SBATCH --job-name=mpi4py-test   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=1G         # memory per cpu-core
#SBATCH --partition=test         # partition
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yitan@g.harvard.edu

module load gcc openmpi
mamba deactivate
mamba activate fast-mpi4py

srun --mpi=pmix -n $SLURM_NTASKS singularity exec /n/home01/ytan/src/dynamite_latest.sif python /n/home01/ytan/deviation_ee/syk/hello_mpi.py
