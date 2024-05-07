import os, subprocess

dir = f'/n/home01/ytan/scratch/deviation_ee/output/collect_state_vecs/'
if not os.path.exists(dir):
   os.makedirs(dir)
file_name = os.path.join(dir, f"collect_state_vecs.sh")
with open (file_name, 'w') as rsh:
    rsh.write(f'''\
#!/bin/bash -l
###### for CPU #####
#SBATCH -J collect_state_vecs        # Job name
#SBATCH --account=yao_lab
#SBATCH --mail-type=END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -o {dir}%j.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
#SBATCH -e {dir}%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1                # Number of tasks
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p shared	    # Partition to submit to

module load python
module load gcc openmpi
mamba deactivate
mamba activate /n/holystore01/LABS/yao_lab/Lab/mamba_envs/complex-opt-0

srun --mpi=pmix -n $SLURM_NTASKS python -O /n/home01/ytan/deviation_ee/syk/collect_state_vecs.py

##### for GPU #####
# #SBATCH -J solve_extrm_eigvals      # Job name
# #SBATCH --account=yao_lab
# #SBATCH --mail-type=END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH -o {dir}%j.out  # File to which STDOUT will be written, %j inserts jobid; dir already ends with / 
# #SBATCH -e {dir}%j.err  # File to which STDERR will be written, %j inserts jobid
# #SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
# #SBATCH -n 1                # Number of tasks
# #SBATCH -N 1                # Ensure that all cores are on one machine
# #SBATCH --mem=32G       # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH -p gpu_requeue,gpu
# #SBATCH --gres=gpu:1
# #SBATCH --constraint="a40|a100"

# mamba deactivate
# singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif python /n/home01/ytan/deviation_ee/syk/collect_state_vecs.py
''')
    rsh.close()
    command = ['sbatch', file_name]
    subprocess.run(command)
    
