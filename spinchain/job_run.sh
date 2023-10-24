#!/bin/bash

for n in 28; do
    # sbatch job_cpu.sbatch --L $n
    sb job_cpu.sbatch --L $n
done

# for ((seed=4; i<=31; i++)); do
#     scancel 53171931_${seed}
# done