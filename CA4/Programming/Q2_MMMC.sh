#!/bin/bash

#SBATCH --job-name=multi_node_accelerate
#SBATCH --partition=partition

#SBATCH --mem=1500mb
#SBATCH --nodes=2

#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=2

##### Number of total processes
echo "----------------------"

echo "Nodelist: " $SLURM_JOB_NODELIST
echo "Number of nodes: " $SLURM_JOB_NUM_NODES

echo "Ntasks per node: " $SLURM_NTASKS_PER_NODE

echo "CPUs per task: 2" 

echo "---------------------- "
INSTANCES_PER_NODE="${INSTANCES_PER_NODE:-1}"

# Master Port
export MASTER_PORT=24442

export RENDEZVOUS_ID=$RANDOM
export WORLD_SIZE=4

### Get the first node name as master address.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_ADDR: $MASTER_ADDR:$MASTER_PORT"
echo "----------------------"

# Just to suppress a warning
export OMP_NUM_THREADS=1

# Activate virtual environment
source /home/shared_files/pytorch_venv/bin/activate


### Accelerate
srun accelerate launch \
 --num_processes=4 \
 --num_machines=$SLURM_NNODES \
 --rdzv_backend c10d \
 --main_process_ip=$MASTER_ADDR \
 --main_process_port=$MASTER_PORT \
 --dynamo_backend=no \
 --mixed_precision=no \
 --machine_rank $SLURM_PROCID \
 --multi_gpu \
 Q2_MMMC.py
