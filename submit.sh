#!/bin/bash
#SBATCH --job-name=mnist_vae
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

python examples/scripts/training.py \
    --dataset mnist \
    --model_name vae \
    --model_config 'examples/scripts/configs/mnist/vae_config.json' \
    --training_config 'examples/scripts/configs/mnist/base_training_config.json'
