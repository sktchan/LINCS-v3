#!/bin/bash

#SBATCH --job-name=v1_rtf
#SBATCH --output=out/v1_rtf.out
#SBATCH --error=out/v1_rtf.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=1-00:00:00

cd /home/golem/scratch/chans/lincsv3
julia scripts/hybrid/version1.jl
