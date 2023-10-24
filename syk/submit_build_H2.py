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
#SBATCH -c {cores} # cpus per task
#SBATCH -t {time} # Maximum execution time (D-HH:MM)
#SBATCH -J {task_idx}_{name}
#SBATCH -p {partition} # Partition
#SBATCH --mem={mem} # Memory
#SBATCH --account {account} 
#SBATCH -o {output_dir}/{task_idx}_%a.out
#SBATCH -e {output_dir}/{task_idx}_%a.err
#SBATCH --open-mode=truncate
#SBATCH --gres=gpu

# for GPU
date
singularity exec --nv /n/holystore01/LABS/yao_lab/Lab/dynamite/sifs/dynamite_latest-cuda.sif \
python /n/home01/ytan/deviation_ee/syk/build_syk_H2_assembly.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)
# python /n/home01/ytan/deviation_ee/syk/build_syk_H2_8uniq.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)
# python /n/home01/ytan/deviation_ee/syk/build_syk_H2_6uniq.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)

# for CPU
# source /n/holystore01/LABS/yao_lab/Lab/dynamite/activate_cpu_64.sh

# date
# srun --mpi=pmix -n {cores} python /n/home01/ytan/scratch/deviation_ee/syk/build_syk_H2_with_trace.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)
# python /n/home01/ytan/scratch/deviation_ee/syk/build_syk_H2_with_trace.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)
'''
program_args = '{size} {seed}'

global_options = {
    # batch options
    'output_dir' : '/n/home01/ytan/scratch/deviation_ee/output_syk/08262023_build_syk_H2', 
    'partition': 'gpu_requeue',
    'account'    : 'yao_lab',
    'mem'    : '8G',

    # program options
    # 'name': 'build_syk_H2_8uniq_L23',
    'name': 'build_syk_H2_assembly_L20',
    'nodes'      : 1,
    'cores'      : 10,
}

# set a custom directory in output_dir for this particular run
global_options['output_dir'] = join(global_options['output_dir'], global_options['name'])

# each of these will be submitted as a separate job array
task_options = [
    {
        'size': 20,
        'seed': range(10),
        # 'time':'3-0:00:00',
        'time': '0-2:00:00',
    },
]

if __name__ == '__main__':
    from submitter import submit
    submit(batch, program_args, global_options, task_options)
