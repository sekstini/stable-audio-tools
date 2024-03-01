#!/bin/bash

set -x

# Submit the first job
job1=$(sbatch launchtest.slurm $1 | cut -d ' ' -f 4)

# Submit subsequent jobs, each dependent on the completion of the previous
job2=$(sbatch --dependency=afterok:$job1 launchtest.slurm --id=$job1 | cut -d ' ' -f 4)
job3=$(sbatch --dependency=afterok:$job2 launchtest.slurm --id=$job1 | cut -d ' ' -f 4)
