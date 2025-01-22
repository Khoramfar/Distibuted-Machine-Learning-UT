#!/bin/bash
#SBATCH --job-name=LogReg_c
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=LogReg_c.out

echo "Job is started"

srun --mpi=pmix_v4 python3 LogReg_c.py

echo "Job is done"
