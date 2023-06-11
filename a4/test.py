import torch 

x = torch.randn(2, 3)

print(x.shape)

print(torch.unsqueeze(x, dim = 2).shape)
