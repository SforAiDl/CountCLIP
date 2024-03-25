#!/bin/sh
#SBATCH --job-name=llm
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=/home/f20220609/llm/output/experiment.out
#SBATCH --error=/home/f20220609/llm/output/experiment.err
#SBATCH --time=1-00:00:00

python experiment.py