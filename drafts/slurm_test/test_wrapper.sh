#!/bin/bash

# Generate a timestamp using shell commands or Python. Slurm does not support variables in the #SBATCH directives, so need this workaround.
# TIMESTAMP=$(date +%Y%m%d)  # Using shell
TIMESTAMP=$(python -c 'from datetime import datetime; print(datetime.now().strftime("%Y%m%d"))')  # Using Python

# Define job name and other identifiers if needed
job_name="test_slurm_io"

# Construct the output and error filenames with the timestamp
output_file="${job_name}_%A_%a_${TIMESTAMP}.out"
error_file="${job_name}_%A_%a_${TIMESTAMP}.err"

# Submit the job with sbatch, specifying the dynamically generated filenames
sbatch --array=0-4 --job-name="$job_name" --output="$output_file" --error="$error_file" test.sh