#!/usr/bin/env python3
# test_conv_cuda.py
import torch
import torch.nn as nn

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Create dummy input (batch=4, channels=1, height=28, width=28)
    x = torch.randn(4, 1, 28, 28, device=device, dtype=torch.float32)

    # Create Conv2d layer
    conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3).to(device)

    # Forward pass
    y = conv(x)

    print("Output shape:", y.shape)
    print("OK")

if __name__ == "__main__":
    main()
