#!/bin/bash
#SBATCH --job-name=sqrt_c_2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=sqrt_c_2.out

echo "Job is started"

srun --mpi=pmix_v4 python3 sqrt_c_2.py

echo "Job is done"
