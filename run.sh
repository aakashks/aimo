#!/bin/bash
#SBATCH --job-name=3c
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=20
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

source ~/.bashrc
z aimo

cat run.sh
cat aimo3c.py
python aimo3c.py
python aimo3c.py
python aimo3c.py
