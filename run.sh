#!/bin/bash
#SBATCH --job-name=t
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=20
#SBATCH --time=8:00:00

source ~/.bashrc
z aimo

FILE="$1"

cat $FILE
python $FILE
python $FILE
