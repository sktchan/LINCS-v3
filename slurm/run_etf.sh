#!/bin/bash

#SBATCH --job-name=exp_tf
#SBATCH --output=out/exp_tf.out
#SBATCH --error=out/exp_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-gpu=32G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=2-00:00:00

# Change to project directory
cd /home/golem/scratch/chans/lincs

# Run the expression transformer script
julia scripts/exp_tf.jl

