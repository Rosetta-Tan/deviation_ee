import os, subprocess, datetime, argparse
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

parser = argparse.ArgumentParser()
parser.add_argument('--dry_run', required=False, action='store_true')
parser.add_argument('--resume', dest='resume', type=int, required=False, default=0)  # 0: False, 1: True
args = parser.parse_args()

# grid submission
Ls = list(range(14, 21, 2))
seeds = list(range(20))
tols = [0.1, 0.01, 0.001]

JOBNAME = 'syk_meas'
dir = '/n/home01/ytan/scratch/deviation_ee/old/output/20240425_powermethod_z2_newtol'
vec_dir = '/n/home01/ytan/scratch/deviation_ee/old/vec_syk_pm_z2_newtol'
if not os.path.isdir(dir):
  os.mkdir(dir)
for L in Ls:
  for tol in tols:
    file_name = os.path.join(dir, f"{JOBNAME}_L={L}_tol={tol}.sh")
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
# #SBATCH -t 0-02:00           # Runtime in D-HH:MM, minimum of 10 minutes
# #SBATCH -n 64                # Number of tasks
# #SBATCH -N 4                # Ensure that all cores are on one machine
# #SBATCH --ntasks-per-node=16
# #SBATCH --mem-per-cpu=8G       # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH -p shared	    # Partition to submit to

# module load python
# module load gcc openmpi
# mamba activate /n/holystore01/LABS/yao_lab/Lab/mamba_envs/complex-opt-0

# srun --mpi=pmix -n $SLURM_NTASKS python -O /n/home01/ytan/deviation_ee/syk_old/solve_syk_powermethod_longtime.py -L {L} --seed $SLURM_ARRAY_TASK_ID --tol {tol} --vec_dir {vec_dir} --log_dir {dir} --duration {24*3600} --resume {args.resume}

##### for GPU #####
#SBATCH -J {JOBNAME}      # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -o {dir}/%A_%a.out        # File to which STDOUT will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -e {dir}/%A_%a.err        # File to which STDERR will be written, %A_%a inserts job master id and array id, $(date +%Y%m%d) inserts date
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem=4G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu_requeue,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk_old/measure_obs.py --L {L} --seed $SLURM_ARRAY_TASK_ID --tol {tol} --vec_dir {vec_dir} --gpu --save True
''')
    rsh.close()
    array_str = ','.join([str(i) for i in seeds])
    filename_str = f'%A_%a_{JOBNAME}_L={L}_seed=%a_{TIMESTAMP}'
    command = ['sbatch', f"--array={array_str}", f"--output={dir}/{filename_str}.out", f"--error={dir}/{filename_str}.err"]
    if L == 26:
        command += ["--time=0-12:00", "--mem=20G"]
    # command += ["--partition=gpu_test", "--time=0-01:00", "--mem=2G", "--constraint="]  # for test purpose
    command += [file_name]
    if args.dry_run:
      print(command)
    else:
      subprocess.run(command)
