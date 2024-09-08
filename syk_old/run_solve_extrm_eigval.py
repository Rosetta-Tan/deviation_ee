import os, subprocess, datetime, argparse
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
JOBNAME = 'solve_extrm_eigvals'

parser = argparse.ArgumentParser()
parser.add_argument('--dry_run', required=False, action='store_true')
args = parser.parse_args()

# grid submission
Ls = list(range(14, 21, 2))
seeds = list(range(0, 20))

# dir = f'/n/home01/ytan/scratch/deviation_ee/old/output/20240403_rerun_solve_extrm_eigvals'
dir = f'/n/home01/ytan/scratch/deviation_ee/old/output/20240516_debug_solve_extrm_eigvals'
res_dir = f'/n/home01/ytan/scratch/deviation_ee/old/extrm_eigval'
if not os.path.isdir(dir):
  os.mkdir(dir)
for L in Ls:
  file_name = os.path.join(dir, f"{JOBNAME}_L={L}.sh")
  with open (file_name, 'w') as rsh:
    rsh.write(f'''\
#!/bin/bash -l
###### for CPU #####
# #SBATCH -J {JOBNAME}        # Job name
# #SBATCH --account=yao_lab
# #SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --chdir={dir}       # Directory for job execution
# #SBATCH -o %A_%a.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
# #SBATCH -e %A_%a.err  # File to which STDERR will be written, %j inserts jobid
# #SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
# #SBATCH -n 64                # Number of tasks
# #SBATCH -N 4                # Ensure that all cores are on one machine
# #SBATCH --ntasks-per-node=16
# #SBATCH --mem-per-cpu=8G       # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH -p shared	    # Partition to submit to

# module load python
# module load gcc openmpi
# mamba deactivate
# mamba activate /n/holystore01/LABS/yao_lab/Lab/mamba_envs/complex-opt-0

# srun --mpi=pmix -n $SLURM_NTASKS python -O /n/home01/ytan/deviation_ee/syk_old/solve_extrm_eigvals.py -L {L} --seed $SLURM_ARRAY_TASK_ID --log_dir {dir}

##### for GPU #####
#SBATCH -J {JOBNAME}      # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --chdir={dir}       # Directory for job execution
#SBATCH -o %A_%a.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
#SBATCH -e %A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -t 0-12:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem=20G       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu,gpu_requeue
#SBATCH --gres=gpu:1
##SBATCH --constraint="a100"

mamba activate qec_numerics
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk_old/solve_extrm_eigval.py --L {L} --seed $SLURM_ARRAY_TASK_ID --gpu --log_dir {dir} --res_dir {res_dir}
''')
  rsh.close()
  array_str = ','.join([str(i) for i in seeds])
  filename_str = f'%A_%a_{JOBNAME}_L={L}_seed=%a_{TIMESTAMP}'
  command = ['sbatch', f"--array={array_str}", f"--output={filename_str}.out", f"--error={filename_str}.err"]
  # command += ["--partition=gpu_test", "--time=0-12:00", "--mem=20G", "--constraint="]  # for test purpose
  command += [file_name]
  if args.dry_run:
    print(command)
  else:
    subprocess.run(command)
