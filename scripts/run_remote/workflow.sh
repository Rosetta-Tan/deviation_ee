#!/bin/bash
L=$1
seed=$2
tol=$3

jid1=$(sbatch build_syk.sh "${L}" "${seed}" | sed 's/Submitted batch job //')
echo "job1 $jid1 submitted"
jid2=$(sbatch --dependency=afterok:$jid1 solve_syk_powermethod.sh | sed 's/Submitted batch job //')
echo "job2 $jid2 submitted"
jid3=$(sbatch --dependency=afterok:$jid2 measure_obs.sh | sed 's/Submitted batch job //')
echo "job3 $jid3 submitted"