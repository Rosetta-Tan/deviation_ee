import os, subprocess, datetime
import argparse
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
JOBNAME = 'build_GA'

parser = argparse.ArgumentParser()
parser.add_argument('--dry_run', required=False, action='store_true')
args = parser.parse_args()

Ls = range(20, 27, 2)
seeds = range(0, 20)

# mapping from L to memory requirement
memory_config = {
  "12": "500M",
  "14": "1G",
  "16": "2G",
  "18": "4G",
  "20": "8G",
  "22": "16G",
  "24": "32G",
  "26": "64G",
}

time_config = {
  "12": "0-0:10:00",
  "14": "0-0:20:00",
  "16": "0-0:40:00",
  "18": "0-1:20:00",
  "20": "2-0:00:00",
  "22": "2-0:00:00",
  "24": "3-0:00:00",
  "26": "3-0:00:00",
}

dir = f'/n/home01/ytan/scratch/deviation_ee/output/20240526_build_GA/'
save_dir = f'/n/home01/ytan/scratch/deviation_ee/msc_npy_GA'
if not os.path.isdir(dir):
  os.mkdir(dir)
for L in Ls:
  memory_requirement = memory_config[str(L)]
  time_requirement = time_config[str(L)]
  LA = L // 2
  file_name = os.path.join(dir, f"build_GA_L={L}_LA={LA}.sh")
  with open (file_name, 'w') as rsh:
    rsh.write(f'''\
#!/bin/bash -l
# ###### for CPU #####
# #SBATCH -J {JOBNAME}             # Job name
# #SBATCH --account=yao_lab
# #SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH -o {dir}%A_%a.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
# #SBATCH -e {dir}%A_%a.err  # File to which STDERR will be written, %j inserts jobid
# #SBATCH -t {time_requirement}          # Runtime in D-HH:MM, minimum of 10 minutes
# #SBATCH -n 1                # Number of tasks
# #SBATCH -N 1                # Ensure that all cores are on one machine
# #SBATCH --ntasks-per-node=1
# #SBATCH --mem={memory_requirement}       # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH -p shared	    # Partition to submit to

# module load python
# module load gcc openmpi
# mamba deactivate
# mamba activate /n/holystore01/LABS/yao_lab/Lab/mamba_envs/complex-opt-0

# srun --mpi=pmix -n $SLURM_NTASKS python -O /n/home01/ytan/deviation_ee/syk/build_GA.py -L {L} --seed $SLURM_ARRAY_TASK_ID --gpu 0

##### for GPU #####
#SBATCH -J {JOBNAME}              # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --chdir={dir}       # Directory for job execution
#SBATCH -o %A_%a.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
#SBATCH -e %A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem={memory_requirement}       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu_requeue,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

module load cuda/12.4.1-fasrc01
mamba activate qec_numerics
python /n/home01/ytan/deviation_ee/syk/build_GA.py --L {L} --LA {LA} \
    --seed $SLURM_ARRAY_TASK_ID --save True --save_dir {save_dir}
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python dynamite_decompose.py --L "$L" --LA "${LA}" --seed 0 --gpu 1
''')
  rsh.close()
  array_str = ','.join([str(i) for i in seeds])
  filename_str = f'%A_%a_{JOBNAME}_L={L}_seed=%a_{TIMESTAMP}'
  command = ['sbatch', f"--array={array_str}", f"--output={filename_str}.out", f"--error={filename_str}.err"]
  # command += ["--partition=gpu_test", "--time=0-0:2:00", "--mem=2G", "--constraint="]  # for test purpose
  command += [file_name]
  if args.dry_run:
    print(command)
  else:
    subprocess.run(command)
