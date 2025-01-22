#!/bin/bash
#SBATCH --job-name=LogReg_b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=LogReg_b.out

echo "Job is started"

srun --mpi=pmix_v4 python3 LogReg_b.py

echo "Job is done"
