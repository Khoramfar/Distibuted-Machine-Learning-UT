#!/bin/bash
#SBATCH --job-name=sqrt_e
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=sqrt_e.out

echo "Job is started"

srun --mpi=pmix_v4 python3 sqrt_e.py

echo "Job is done"
