import torch 

version = torch.__version__
print(version)

version2 = torch.version.cuda
print(version2)

gpu_check = torch.cuda.is_available()
print(gpu_check)