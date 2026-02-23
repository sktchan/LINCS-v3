#!/bin/bash

#SBATCH --job-name=exp_tf
#SBATCH --output=out/exp_tf.out
#SBATCH --error=out/exp_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=1-00:00:00

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/golem/scratch/chans/lincs
julia scripts/exp_tf.jl

