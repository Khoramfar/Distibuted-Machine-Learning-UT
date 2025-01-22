#!/bin/bash
#SBATCH --job-name=sqrt_d_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=sqrt_d_1.out

echo "Job is started"

srun --mpi=pmix_v4 python3 sqrt_d_1.py

echo "Job is done"
