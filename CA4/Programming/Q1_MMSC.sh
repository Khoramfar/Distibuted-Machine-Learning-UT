#!/bin/bash

#SBATCH --job-name=multi_node_pytorch_torchrun
#SBATCH --partition=partition

#SBATCH --mem=1000mb
#SBATCH --nodes=2

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

##### Number of total processes
echo "----------------------"

echo "Nodelist: " $SLURM_JOB_NODELIST
echo "Number of nodes: " $SLURM_JOB_NUM_NODES

echo "Ntasks per node: " $SLURM_NTASKS_PER_NODE

echo "CPUs per task: 1" 

echo "---------------------- "

# Master Port
export MASTER_PORT=24442

export RENDEZVOUS_ID=$RANDOM
export WORLD_SIZE=2

### Get the first node name as master address.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_ADDR: $MASTER_ADDR:$MASTER_PORT"
echo "----------------------"

# Just to suppress a warning
export OMP_NUM_THREADS=1

# Activate virtual environment
source /home/shared_files/pytorch_venv/bin/activate


### Torchrun
srun torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=$RENDEZVOUS_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT Q1_MMSC.py