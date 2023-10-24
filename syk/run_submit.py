
import argparse as ap
from subprocess import check_output
from os import path,mkdir
from expand_vals import expand_vals
from copy import deepcopy
from time import sleep
from params import vals_default, vals_mod, batch

vals = []
for d in vals_mod:
    vals.append(deepcopy(vals_default))
    vals[-1].update(d)

parser = ap.ArgumentParser()
parser.add_argument('--dry-run',help='output batch script but do not submit',
                    action="store_true")
args = parser.parse_args()

# make sure we won't overwrite
if path.exists(vals_default['output_dir']) and not args.dry_run:
    if not input('output directory exists, overwrite? ') in ['y','Y']:
        exit()

for n,d in enumerate(expand_vals(vals)):

    # be nice to the scheduler
    if n > 0:
        sleep(0.1)
    
    # append job index to the job name
    # d['name'] = '{:03d}_'.format(n) + d['name']
    d['job_idx'] = '{:03d}'.format(n)

    batch_out = batch.format(**d).format(**d)

    if args.dry_run:
        print(batch_out)
    else:
        # create output directory if it doesn't exist
        out_path = d['output_dir'].rstrip('/')
        to_make = []
        while out_path and not path.exists(out_path):
            to_make.append(out_path)
            out_path = path.split(out_path)[0]
        for p in to_make[::-1]:
            mkdir(p)
            
        with open(path.join(d['output_dir'],d['job_idx'])+'.batch','w') as f:
            f.write(batch_out)

        print(check_output(['sbatch'],input=batch_out,universal_newlines=True),end='')
