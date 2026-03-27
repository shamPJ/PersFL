#!/bin/bash
#SBATCH --job-name=test_conv
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_conv_%j.out
#SBATCH --error=logs/test_conv_%j.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env-cuda118

# ===============================
# Create logs directory
# ===============================
mkdir -p logs

# ===============================
# Run minimal CUDA Conv2d test
# ===============================
srun python - << 'EOF'
import torch
import torch.nn as nn

# torch.backends.cudnn.enabled = False  # disable cuDNN engine

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled:", torch.backends.cudnn.enabled)

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Dummy input
    x = torch.randn(4, 1, 28, 28, device=device, dtype=torch.float32)

    # Conv2d layer
    conv = nn.Conv2d(1, 8, 3).to(device)

    # Forward pass
    y = conv(x)
    print("Output shape:", y.shape)
    print("OK")

if __name__ == "__main__":
    main()
EOF
