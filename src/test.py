import torch

inputlabel = torch.randint(0, 30, [32])
labels = torch.full([32, 30], 0)
noise = torch.randn(32, 64)
for index in range(0, len(inputlabel)):
    labels[index][inputlabel[index]] = 1

print(torch.cat((labels, noise), 1).size())
