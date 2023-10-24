#!/usr/bin/python3

"""
submit batch jobs to the queue
"""

from subprocess import check_output
from functools import reduce
from itertools import product, repeat
from copy import deepcopy
from time import sleep
from os import path,mkdir
from sys import argv,exit
import numpy as np

## SIMULATION PARAMETERS
Ly = 5 #g
num_chis = 4 #g
num_int_sweeps = 30 #g
scramble_initial_state = 'False' #g

## JOB PARAMETERS
jname = 'syk%d'%(L)
jaccount = "yao_lab"
jpartition = "shared"
jtotal_time = "0-00:30:00"
jmemory = '8G'
jcores = 48

## EDIT  DIRECTORIES AND FILES
SCRATCH_DIR ='/n/holyscratch01/yao_lab/Rahul/'
HOME_DIR = '/n/holystore01/LABS/yao_lab/Everyone/rsahay/LSSC/'
OUTPUT_DIR = SCRATCH_DIR + 'LSSC/InLayerU/'
MODULE_FILE = HOME_DIR + 'InLayerU/thing.sh'
TENPY_DIR =  HOME_DIR + 'git_SC_SW_insulator/tenpy'
PYTHON_FILE = HOME_DIR + 'git_SC_SW_insulator/DMRG/Simulation.py'

assert path.isdir(SCRATCH_DIR), 'Scratch Dir: %s does not exist'%(SCRATCH_DIR)
assert path.isdir(HOME_DIR), 'Home Dir: %s does not exist'%(HOME_DIR)
assert path.isdir(OUTPUT_DIR), 'Output Dir: %s does not exist'%(OUTPUT_DIR)
assert path.isfile(MODULE_FILE), 'Module File: %s does not exist'%(MODULE_FILE)
assert path.isdir(TENPY_DIR), 'TenPy Dir: %s does not exist'%(TENPY_DIR)
assert path.isfile(PYTHON_FILE), 'Python File: %s does not exist'%(PYTHON_FILE)

#print('output subdirectory name: ')
#temp = raw_input()
OUTPUT_DIR = path.join(OUTPUT_DIR, input("output subdirectory name: "))

if '-dry-run' not in argv:
    if not path.isdir(OUTPUT_DIR):
        mkdir(OUTPUT_DIR)
    else:
        #print("directory exists. overwrite? (y/n)")
        #ans = raw_input()
        if input("directory exists. overwrite? (y/n)") not in ['y','Y']:
            exit()

def get_lists(keys_so_far, d):
    sub_lists = []
    for key, val in d.items():
        if isinstance(val, list):
            sub_lists.append( [keys_so_far+[key],val] )
        elif isinstance(val, dict):
            sub_lists += get_lists(keys_so_far+[key],val)
        else:
            sub_lists.append( [keys_so_far+[key],[val]] )
    return sub_lists

def make_product( d ):
    '''
    Take a dictionary and return the product of options in the lists.
    I know that didn't make much sense. it does in my head.
    '''
    lsts = get_lists([],d)

    return product(*[zip(repeat(x[0]),x[1]) for x in lsts])

def get_nested_dict( d, key_list ):
    return reduce(dict.__getitem__,key_list,d)

#'serial_requeue'
#<---------------------EDIT THIS -------------------->
g_list = [0.25]#[0.1, 0.13, 0.15, 0.2, 0.25] #g
US_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0] #g
UD_list = [0]

options = {
    'batch_options' : {
        '-J':'%s'%(jname),
        '--account=%s'%(jaccount): '',
        '--partition=%s'%(jpartition): '',
        '-n': '1',
        '--nodes=1': '',
        '--priority':'TOP',
	'-c':'%s'%(jcores),
        '--time=%s'%(jtotal_time): '',
        #'--mem-per-cpu':'%s'%(jmemory),
        #'--constraint="k40m"': '', #NEEDED FOR L24
        #'--constraint="k40m|k20m|k80|k20xm"': '', 
        #'--gres=gpu':'',
        '--contiguous':'',
        '--open-mode=appen': '',
    },

    'run_options' : {
        'output_dir' : OUTPUT_DIR + "/",
        'Ly' : '%d'%(Ly),
        'Lx' : '4',
        'q'  : '4',
        'factor' : '1',
        'chi': '%d'%(chi),
        'nu' : '0.5',
        'num_chis': '%d'%(num_chis),
        'num_int_sweeps': '%d'%(num_int_sweeps),
        'spinful': 'True',
        'opposite_chern': 'True',
        'normal_ordered': 'False',
        'scramble_initial_state': '%s'%(scramble_initial_state),
        'g' : [str(i) for i in g_list],
        'US': [str(i) for i in US_list], #US = (U + V)/2                                                                     
        'UD': [str(i) for i in UD_list]  #UD = (U - V)/2
    }
}

print(options['batch_options'])
print(options['run_options'])
#exit()

# output these options so that we remember what they were
if '-dry-run' not in argv:
    with open(path.join(OUTPUT_DIR,'run_options'),'w') as f:
        f.write('batch_options:\n')
        for opt,val in options['batch_options'].items():
            f.write('   '+str(opt)+':'+str(val)+'\n')
        f.write('run_options:\n')
        for opt,val in options['run_options'].items():
            f.write('   '+str(opt)+':'+str(val)+'\n')
        
run_lst = []
for n,x in enumerate(make_product(options)):
    tmp_d = {'batch_options':{},'run_options':{}}
    for key_list,value in x:
        if isinstance(value,dict): # don't clobber other stuff in a dict there
            get_nested_dict(tmp_d,key_list[:-1])[key_list[-1]].update(value)
        else: # this is just a regular value
            get_nested_dict(tmp_d,key_list[:-1])[key_list[-1]] = value

    # give the job a unique name and places to output stuff
    tmp_d['batch_options']['--output'] = path.join(OUTPUT_DIR, '%a_'+tmp_d['batch_options']['-J']+'.out')
    tmp_d['batch_options']['-e'] = path.join(OUTPUT_DIR, '%a_'+tmp_d['batch_options']['-J']+'.err')
    tmp_d['batch_options']['-J'] = OUTPUT_DIR.split('/')[-1]+'_'+tmp_d['batch_options']['-J']

    run_lst.append(tmp_d)

for job_n,d in enumerate(run_lst):

    s = ''
    for opt,val in d['run_options'].items():
        #print(opt,val)
        s += ','.join([opt,val])+'\n'

    if not '-dry-run' in argv:
        print(path.join(OUTPUT_DIR,str(job_n))+'.opts')
        with open(path.join(OUTPUT_DIR,str(job_n))+'.opts','w') as f:
            f.write(s)
    else:
        print(job_n)
        print(s)
        print()
batch_script = '#!/bin/bash'

print(d['batch_options'])
for opt,val in d['batch_options'].items():
    if 'iojjioj' in opt:
        print('What')
    batch_script += ' '.join(['\n#SBATCH',opt,val])


#<---------------------- EDIT THIS !!!!!! ------------------------------>#
batch_script += '\nsource %s'%(MODULE_FILE)
batch_script += '\nexport PYTHONPATH=%s'%(TENPY_DIR)
batch_script += '\nexport OMP_NUM_THREADS='+str(d['batch_options']['-c'])+' '
batch_script += '\nexport MKL_NUM_THREADS='+str(d['batch_options']['-c'])+' '
batch_script += '\nexport MKL_DYNAMIC=FALSE'
batch_script += '\nsrun --mpi=pmi2 -n '+str(d['batch_options']['-n'])+' -c '+str(d['batch_options']['-c'])
batch_script += ' python %s '%(PYTHON_FILE)
batch_script += path.join(OUTPUT_DIR,'${SLURM_ARRAY_TASK_ID}.opts')

if '-dry-run' in argv:
    print(batch_script)
else:
    # write out the batch script to keep track of what we did
    with open(path.join(OUTPUT_DIR,d['batch_options']['-J'])+'.batch','w') as f:
        f.write(batch_script)
        
    print(check_output(['sbatch','--array=0-'+str(len(run_lst)-1)],
                       input=batch_script,
                       universal_newlines=True),
          '('+str(len(run_lst))+')')

