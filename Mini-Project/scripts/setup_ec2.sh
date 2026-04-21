#!/bin/bash
# Setup script for EC2 Deep Learning AMI

set -e

echo "Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install ogb wandb pyyaml scikit-learn tqdm

echo ""
echo "Login to wandb for experiment tracking:"
wandb login

echo ""
echo "Pre-downloading datasets..."
python scripts/download_data.py

echo ""
echo "GPU check:"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo ""
echo "Setup complete"
