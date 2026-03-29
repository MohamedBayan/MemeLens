#!/bin/bash -l
#SBATCH -J memelens
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH -p <your-gpu-partition>
#SBATCH -A <your-account>
#SBATCH -q <your-qos>
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=30G

module load cuda12.2/toolkit/12.2.1 slurm
source $HOME/anaconda3/bin/activate
conda activate memelens

# Uncomment the script you want to run:
# bash training/memelens/train_stage1_classification.sh
# bash training/memelens/train_stage2_explanation.sh
# bash inference/run_memelens.sh
# bash inference/run_zero_shot.sh
