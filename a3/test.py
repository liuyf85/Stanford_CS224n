import torch

# Create a tensor of zeros
x = torch.empty([3, 4])

# Add some data to the tensor
x[0] = torch.tensor([1, 2, 3, 4])
x[1] = 2
x[2, 1] = 3

print(x)
