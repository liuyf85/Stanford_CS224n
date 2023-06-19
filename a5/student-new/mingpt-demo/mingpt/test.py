import torch

x = torch.tensor([i for i in range(12)]).view(3, 4)
print(x)

# print(torch.tensor([1, 2, 3])[:2])

print(x[ :  ][ : 2 ])
print(x[ :  , 1: 2 ])