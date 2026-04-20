#!/bin/bash
# Setup script for EC2 Deep Learning AMI (Ubuntu)
# Run once after launching the instance:
#   bash scripts/setup_ec2.sh

set -e

echo "=== Setting up experiment environment ==="

# 1. Install PyG and dependencies
echo "Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install ogb wandb pyyaml scikit-learn tqdm

# 2. Login to wandb (interactive)
echo ""
echo "Login to wandb for experiment tracking:"
wandb login

# 3. Pre-download datasets
echo ""
echo "Pre-downloading datasets..."
python scripts/download_data.py

# 4. Verify GPU access
echo ""
echo "GPU check:"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo ""
echo "=== Setup complete ==="
echo "To run experiments:"
echo "  tmux new -s experiments"
echo "  python scripts/launch_all.py --all"
echo "  # Ctrl+b d to detach"
