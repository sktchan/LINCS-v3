#!/bin/bash

#SBATCH --job-name=rank_ae
#SBATCH --output=out/rank_ae.out
#SBATCH --error=out/rank_ae.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=5-00:00:00

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/golem/scratch/chans/lincsv3
julia scripts/rank_tf.jl
