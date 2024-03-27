#!/bin/sh

#SBATCH --job-name=temp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

python3 runner.py