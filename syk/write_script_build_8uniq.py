import argparse,os

parser = argparse.ArgumentParser(description='Feed in parameters for the run')
parser.add_argument('-L', type=int, required=True, help='system size')
# parser.add_argument('--nseeds', type=int, required=False, nargs='?', help='sample seed') # nargs='?' means 0 or 1 argument
args = parser.parse_args()
L = args.L
# nseeds = args.nseeds

dir = f'/n/home01/ytan/scratch/deviation_ee/output/build_8uniq/'
if not os.path.isdir(dir):
  os.mkdir(dir)
for seed in range(10):  # 10 seeds
  file_name = os.path.join(dir, f"L={L}_seed={seed}.sh")
  with open (file_name, 'w') as rsh:
    rsh.write(f'''\
#!/bin/bash -l
#SBATCH -J syk              # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -n 1                # Number of tasks
#SBATCH -c 16               # Number of threads per task
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue,shared	    # Partition to submit to
#SBATCH --mem-per-cpu=8G          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o {dir}%j_build_8uniq.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
#SBATCH -e {dir}%j_build_8uniq.err  # File to which STDERR will be written, %j inserts jobid

module load python
module load gcc openmpi
mamba deactivate
mamba activate /n/holystore01/LABS/yao_lab/Lab/mamba_envs/complex-opt-0

srun --mpi=pmix -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK python -O /n/home01/ytan/deviation_ee/syk/build_syk_H2_8uniq.py -L {L} --seed {seed}
''')