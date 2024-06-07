#!/bin/bash
#SBATCH --job-name s1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=20
#SBATCH --time=6:00:00

source ~/.bashrc
conda activate temp
z aimo

papermill 13.ipynb 13.ipynb
