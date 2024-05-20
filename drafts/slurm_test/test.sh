#!/bin/bash -l
#SBATCH --job-name=test_slurm_io # create a name for your job
#SBATCH -e "test_slurm_io_%A_%a_${TIMESTAMP}.err" # error file
#SBATCH -o "test_slurm_io_%A_%a_${TIMESTAMP}.out" # output file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=1G         # memory per cpu-core
#SBATCH --partition=test         # partition
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yitan@g.harvard.edu

# This will subtitute both the original text and the output text
python print_smt.py | sed 's/place/spot/g' | tee output.txt


# This will keep the original text unchanged and only substitute the output text
# Run your Python script and store the output in a temporary file
python print_smt.py > tmp_output.txt

# Output the original text to the Slurm output file
cat tmp_output.txt

# Perform the substitution and write the modified text to another file
sed 's/place/spot/g' tmp_output.txt > output.txt

# Clean up the temporary file
rm tmp_output.txt
