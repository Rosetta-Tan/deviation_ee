
from copy import deepcopy
from time import sleep
from itertools import product
import argparse as ap
from subprocess import check_output
from os import path, mkdir

def submit(batch, program_args, global_options, task_options):
    '''
    Submit a job to the batch system.

    Parameters
    ----------

    batch : str
        A string containing the framework of the batch script, with substitutable fields in
        curly braces.

        Fields that should be defined:
         - "name", which is appended to with the task number

        Extra fields which can be specified are:
         - "task_idx", incremented for each task submitted

    program_args : string or list
        A string representing the command line options, with substitutable fields in braces.
        If a list is passed, a CSV file will be written with the options.

    global_options : dict
        A dict containing keys corresponding to the fields in curly braces, with the default
        options for all tasks

    task_options : dict
        A dict with keys corresponding to the parameters in curly braces, to be substituted
        for each batch task submitted

    Returns
    -------
    '''

    for k in global_options:
        # print('k', k)
        if isinstance(program_args, str):
            program_arg_str = program_args
            # print('program_arg_str', program_arg_str)
        else:
            program_arg_str = ''.join('{%s}'%x for x in program_args)
            # print('program_arg_str', program_arg_str)

        if '{'+str(k)+'}' not in batch + program_arg_str:
            print('Warning: key %s from global_options not used.' % k)

    
    out_path = global_options['output_dir'].rstrip('/')
    to_make = []
    while out_path and not path.exists(out_path):
        to_make.append(out_path)
        out_path = path.split(out_path)[0]

    if len(to_make) == 0 and not input('output directory exists, overwrite? ') in ['y','Y']:
        exit()

    for p in to_make[::-1]:
        mkdir(p)
    
    job_idx = 0
    for n,d in enumerate(task_options):
        print('n', n, 'd', d)
        # be nice to the scheduler
        if n > 0:
            sleep(1)

        tmp_vals = deepcopy(global_options)
        tmp_vals.update(d)
        tmp_vals.update({'task_idx': '%03d' % n})

        # also make sure we don't overwrite the actual curly braces we want
        # (this is kind of ridiculous)
        tmp_vals['SLURM_ARRAY_TASK_ID'] = '{SLURM_ARRAY_TASK_ID}'

        # batch_out = batch.format(**tmp_vals).format(**tmp_vals)
        print(tmp_vals)
        batch_out = batch.format(**tmp_vals)

        expanded = expand_vals([tmp_vals])
        print(expanded)
        for nv,v in enumerate(expanded):

            v.update({'array_idx':nv})
            v.update({'job_idx':job_idx})

            job_idx += 1

            if isinstance(program_args, str):
                o = program_args.format(**v)
            else:
                o = ''
                for option_name in program_args:
                    o += '%s,%s\n' % (option_name, v[option_name])

            with open(path.join(tmp_vals['output_dir'],tmp_vals["task_idx"]+'_'+str(nv)+'.opts'),'w') as f:
                f.write(o)

        nv = len(expanded)

        # command = ['sbatch','--array=0-%d' % (nv-1)]
        # print(check_output(command, input=batch_out, universal_newlines=True), end='')
        print('(%d tasks)' % nv)
    

def expand_vals(vals):
    '''
    Take dictionary or a list of dictionaries, for which some values may be lists, and expand
    them as a list of dictionaries for which the values are taken from the "tensor product" of
    the lists.
    '''

    if isinstance(vals,dict):
        vals = [vals]

    out_dicts = []
    for d in vals:
        l = []
        for k,v in d.items():
            if isinstance(v,str):
                v = [v]

            try:
                v = iter(v)
            except TypeError:
                v = [v]

            l.append([(k,str(x)) for x in v])

        p = product(*l)
        for val_set in p:
            out_dicts.append(dict(val_set))

    return out_dicts


if __name__ == '__main__':
    batch = '''#!/bin/bash -l
#SBATCH -N {nodes} # number of nodes
#SBATCH -n {tasks} # ntasks
#SBATCH -t {time} # Maximum execution time (D-HH:MM)
#SBATCH -J {name}_{task_idx}
#SBATCH -p {partition} # Partition
#SBATCH --mem-per-cpu={mem}
#SBATCH --account {account} 
#SBATCH -o {output_dir}/{task_idx}_%a.out
#SBATCH -e {output_dir}/{task_idx}_%a.err
#SBATCH --open-mode=truncate

source /n/holystore01/LABS/yao_lab/Lab/dynamite/activate_cpu.sh
date
srun --mpi=pmix -n {tasks} python /n/home01/ytan/scratch/deviation_ee/syk/build_syk_H2_with_trace.py $(cat {output_dir}/{task_idx}_${SLURM_ARRAY_TASK_ID}.opts)
'''
    program_args = '{size} {seed}'

    global_options = {
        # batch options
        'output_dir' : '/n/home01/ytan/scratch/deviation_ee/output_syk/09132023_build_syk_H2_assemble_terms', 
        'partition': 'cpu_requeue',
        'account'    : 'yao_lab',
        'mem'    : '8G',

        # program options (real run)
        'name': 'debug_submitter',
        'nodes'      : 2,
        'tasks'      : 4,
    }

    # set a custom directory in output_dir for this particular run
    global_options['output_dir'] = path.join(global_options['output_dir'], global_options['name'])

    # each of these will be submitted as a separate job array
    task_options = [
        {
        'size': [1, 2],
        'seed': 1,
        'time':'1-00:00',
        }
    ]   
    submit(batch, program_args, global_options, task_options)