import torch

print(f"CUDA(nVidia): {torch.cuda.is_available()}")
print(f"ROCm (AMD): {torch.version.hip}")
print(f"MPS (Mac): {torch.backends.mps.is_available()}")
