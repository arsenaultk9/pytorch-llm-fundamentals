#!/bin/bash 
#SBATCH --job-name=torch_train_lstm_kears4
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --partition=gpu_96h 
#SBATCH --cpus-per-task=4 
#SBATCH --time=0-01:00 
#SBATCH --mem=16G

module load StdEnv/2020 python/3.10.2 gcc/9.3.0 cuda/11.4 arrow/13.0.0

module list
avail_wheels "torch*"

pip install -r valeria_requirements.txt
pip freeze
python -u train.py