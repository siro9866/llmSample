import torch

if torch.backends.mps.is_available():
    print("✅ MPS (Apple GPU) 사용 가능")
else:
    print("❌ MPS 사용 불가")