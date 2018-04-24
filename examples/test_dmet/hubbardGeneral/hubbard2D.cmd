#!/bin/bash
# parallel job using 20 processors. and runs for 1 hours (max)

#SBATCH --qos=heavy
#SBATCH --partition=serial
#SBATCH -N 1
#SBATCH --ntasks-per-node=14
#SBATCH -c 2
#SBATCH -t 100:00:00
#SBATCH --mem=256000

export SLURM_MPI_TYPE=pmi2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

srun hostname |sort
export SCRATCH="/scratch/global/zhendong/hubbard2D/"
mkdir -p $SCRATCH
rm -r $SCRATCH
mkdir -p $SCRATCH
srun python -u main.py > main.out
rm -r $SCRATCH
