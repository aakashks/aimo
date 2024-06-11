#!/bin/bash
#SBATCH --job-name=3n
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=20
#SBATCH --time=8:00:00

source ~/.bashrc
conda activate temp
z aimo

FILE="aimo3n.py"

cat $FILE
python $FILE
python $FILE
