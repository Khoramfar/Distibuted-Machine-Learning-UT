#!/bin/bash
#SBATCH --job-name=LogReg_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=partition
#SBATCH --output=LogReg_a.out

echo "Job is started"

srun --mpi=pmix_v4 python3 LogReg_a.py

echo "Job is done"
