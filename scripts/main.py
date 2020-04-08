import torch 

t = torch.tensor([1,2,3])

gpu = t.cuda()
print(gpu)