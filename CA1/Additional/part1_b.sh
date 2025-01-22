#!/bin/bash
#SBATCH --job-name=part1_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=partition
#SBATCH --output=part1_b.out

echo "Job is started"

srun --mpi=pmix_v4 python3 part1_b.py

echo "Job is done"
