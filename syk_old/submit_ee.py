from os.path import expanduser,join
import numpy as np
from copy import deepcopy

# the batch script
# important parts:
#
# - for output and error files, remember to include %a to get output from each array run
# - the options are stored in "{output_dir}/{run_idx}_$SLURM_ARRAY_TASK_ID.opts".

batch = '''#!/bin/bash -l
#SBATCH -N {nodes}
#SBATCH -c {cores}
#SBATCH -t {time}
#SBATCH -J {task_idx}_{name}
#SBATCH -p {partition} # Partition
#SBATCH --mem={mem} # Memory
#SBATCH --account {account}
#SBATCH -t {time} # Maximum execution time (D-HH:MM)
#SBATCH -o {output_dir}/{task_idx}_%a.out
#SBATCH -e {output_dir}/{task_idx}_%a.err
#SBATCH --open-mode=truncate
#SBATCH --gres=gpu

# for GPU
date
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif \
python /n/home01/ytan/deviation_ee/syk/run_ee_eig_H2_gs_noparity.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)

# for CPU
# source /n/holystore01/LABS/yao_lab/Lab/dynamite/activate_cpu_64.sh

# date
# # srun --mpi=pmix -n {cores} python /n/home01/ytan/deviation_ee/syk/run_ee_eig.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)
# python /n/home01/ytan/deviation_ee/syk/run_ee_eig.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)
'''
program_args = '-L {size} -m {msc_dir} --outfile {output_dir}/{task_idx}_{array_idx} --H_idx {H_idx} -s'

global_options = {
    # batch options
    'output_dir' : '/n/home01/ytan/scratch/deviation_ee/output_syk/09112023_run_ee_eig_H2_gs_noparity', 
    'partition': 'gpu_requeue',
    'account'    : 'yao_lab',
    'mem'    : '20G',

    # program options
    'name': 'run_ee_eig_H2_gs_L20',
    'msc_dir'    : '/n/home01/ytan/scratch/deviation_ee/msc_syk',
    'nodes'      : 1,
    'cores'      : 1,
}

# set a custom directory in output_dir for this particular run
global_options['output_dir'] = join(global_options['output_dir'], global_options['name'])

# each of these will be submitted as a separate job array
task_options = [
    {
        # 'name':'L15',
        'size': 20,
        'time':'0-2:00:00',
        'H_idx':range(10),
    },
]

if __name__ == '__main__':
    from submitter import submit
    submit(batch, program_args, global_options, task_options)
