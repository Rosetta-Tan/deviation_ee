import argparse,os

parser = argparse.ArgumentParser(description='Feed in parameters for the run')
parser.add_argument('-Ls', type=int, nargs='+', required=True, help='system size')  # nargs='+' means 1 or more arguments
# parser.add_argument('--nseeds', type=int, required=False, nargs='?', help='sample seed') # nargs='?' means 0 or 1 argument
args = parser.parse_args()
Ls = args.Ls
# nseeds = args.nseeds

dir = f'/n/home01/ytan/scratch/deviation_ee/output/solve_polyfilter/'
if not os.path.isdir(dir):
  os.mkdir(dir)
for L in Ls:
  for seed in range(10):  # 10 seeds
    file_name = os.path.join(dir, f"L={L}_seed={seed}.sh")
    with open (file_name, 'w') as rsh:
      rsh.write(f'''\
#!/bin/bash -l
###### for CPU #####
# #SBATCH -J syk              # Job name
# #SBATCH --account=yao_lab
# #SBATCH --mail-type=END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH -o {dir}%j.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
# #SBATCH -e {dir}%j.err  # File to which STDERR will be written, %j inserts jobid
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

# srun --mpi=pmix -n $SLURM_NTASKS python -O /n/home01/ytan/deviation_ee/syk/solve_syk_polyfilter.py -L {L} --seed {seed}

##### for GPU #####
#SBATCH -J syk              # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -o {dir}%j.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
#SBATCH -e {dir}%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem=32G       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu_requeue,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

mamba deactivate
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/solve_syk_polyfilter.py -L {L} --seed {seed}              
''')
