#!/bin/bash

#SBATCH --job-name=exp_ae
#SBATCH --output=out/exp_ae.out
#SBATCH --error=out/exp_ae.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=1-00:00:00

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/golem/scratch/chans/lincsv3
julia scripts/exp_ae.jl
