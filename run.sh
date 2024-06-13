#!/bin/bash
#SBATCH --job-name=3b
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=20
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

source ~/.bashrc
z aimo

cat run.sh
cat aimo3b.py
python aimo3b.py --cfile code21.py --temp 0.8
python aimo3b.py --cfile code22.py --temp 0.8
