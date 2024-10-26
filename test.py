import torch


if torch.cuda.is_available():
    print("CUDA is available!")
    print("Device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
else:
    print("CUDA is not available.")
