#!/bin/bash
#SBATCH --job-name=t
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=20
#SBATCH --time=8:00:00

source ~/.bashrc
z aimo

cat aimo3b1.py
python aimo3b1.py --cfile code13.py --temp 0.8
python aimo3b1.py --cfile code14.py --top_p 0.99 --temp 0.9
