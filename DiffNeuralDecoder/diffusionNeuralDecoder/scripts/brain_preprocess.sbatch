#!/bin/bash
#SBATCH --job-name=brain_preprocess
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --output=logs/preprocess%j.out
#SBATCH --error=logs/preprocess%j.err

eval "$(micromamba shell hook --shell bash)"
micromamba activate neuralDecoder_env

mkdir -p logs 

cd /home/afan2025/Masters/Deep_Learning_Systems/CSMC_35200_BraintoTextGroup/DiffNeuralDecoder/diffusionNeuralDecoder

python scripts/preprocess_brain.py