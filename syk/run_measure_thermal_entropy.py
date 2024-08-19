import os, subprocess, datetime, argparse
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

parser = argparse.ArgumentParser()
parser.add_argument('--dry_run', required=False, action='store_true')
parser.add_argument('--resume', dest='resume', type=int, required=False, default=0)  # 0: False, 1: True
args = parser.parse_args()

# grid submission
Ls = [14]
seeds = [7,8]
tols = [0.01]

JOBNAME = 'syk_thermal'
dir = '/n/home01/ytan/scratch/deviation_ee/output/2024602_thermal_bound'
vec_dir = '/n/home01/ytan/scratch/deviation_ee/vec_syk_pm_z2_newtol'
if not os.path.isdir(dir):
  os.mkdir(dir)
for L in Ls:
  for tol in tols:
    file_name = os.path.join(dir, f"{JOBNAME}_L={L}_tol={tol}.sh")
    with open (file_name, 'w') as rsh:
      rsh.write(f'''\
#!/bin/bash -l
##### for CPU #####
# #SBATCH -J {JOBNAME}        # Job name
# #SBATCH --account=yao_lab
# #SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --chdir={dir}       # Directory for job execution
# #SBATCH -o %A_%a.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
# #SBATCH -e %A_%a.err  # File to which STDERR will be written, %j inserts jobid
# #SBATCH -t 0-02:00           # Runtime in D-HH:MM, minimum of 10 minutes
# #SBATCH -N 1                # Ensure that all cores are on one machine
# #SBATCH --ntasks-per-node=1
# #SBATCH --mem-per-cpu=8G       # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH -p test	    # Partition to submit to

# module load python
# module load gcc openmpi
# mamba activate /n/holystore01/LABS/yao_lab/Lab/mamba_envs/complex-opt-0

# srun --mpi=pmix python -O /n/home01/ytan/deviation_ee/syk/measure_thermal_entropy.py --L {L} --seed $SLURM_ARRAY_TASK_ID --tol {tol} --save True

##### for GPU #####
#SBATCH -J {JOBNAME}      # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -o {dir}/%j.out        # File to which STDOUT will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -e {dir}/%j.err        # File to which STDERR will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem=20G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu_requeue,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"

singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/measure_thermal_entropy.py --L {L} --LA {L//2} --seed $SLURM_ARRAY_TASK_ID --tol {tol} --gpu --save True
''')
    rsh.close()
    filename_str = f'%A_%a_{JOBNAME}_L={L}_tol={tol}_{TIMESTAMP}'
    array_str = ','.join([str(i) for i in seeds])
    command = ['sbatch', f"--array={array_str}", f"--output={dir}/{filename_str}.out", f"--error={dir}/{filename_str}.err"]
    # command += ["--partition=gpu_test", "--time=0-01:00", "--mem=2G", "--constraint="]  # for test purpose
    command += [file_name]
    if args.dry_run:
      print(command)
    else:
      subprocess.run(command)
