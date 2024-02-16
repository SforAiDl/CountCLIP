#!/bin/sh
#SBATCH --job-name=temp
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --mem=32GB
#SBATCH --output=temp.out
#SBATCH --error=temp.err
#SBATCH --time=1-00:00:00

# activate virtualenv
conda activate mlrc

# execute
torchrun --standalone --nproc_per_node=gpu experiment.py