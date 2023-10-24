#!/bin/bash
#SBATCH --array=0-4
##SBATCH -J mixfld_ising_dnm_imagevol
#SBATCH -J syk_dnm_imagevol
#SBATCH -t 0-2:00:00
#SBATCH -o out/05-08-2023-mixfld_ising_dnm_imagevol/%A-%a.out
#SBATCH -e out/05-08-2023-mixfld_ising_dnm_imagevol/%A-%a.err
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --open-mode=append

# special case
# python src/special_case.py power_method --L $SLURM_ARRAY_TASK_ID

# power method (tensorsnet)
# python pwmthd/pwmthd.py --L $SLURM_ARRAY_TASK_ID

# local chatoc Hamiltonian
# python src/local_chaotic.py power_method --L $SLURM_ARRAY_TASK_ID

# dynamite
source /n/home08/gdmeyer/dynamite/activate_cpu.sh
# source /n/home08/gdmeyer/dynamite/activate_cpu64.sh
srun --mpi=pmix python src/dnm_imagevol.py --L 14 -s $SLURM_ARRAY_TASK_ID --shell