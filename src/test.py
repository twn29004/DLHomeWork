import torch
from torch import nn
from torch.autograd import Variable


class NetG(nn.Module):
    def __init__(self, lenZ):
        super(NetG, self).__init__()
        self.len = lenZ
        # self.convtrav = nn.Conv2d(1, 64, 5, 2, 2, bias=False)s
        self.main = nn.Sequential(
            nn.Linear(4096, 30),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        # 首先对输入进行reshape
        # input = input.reshape((1, self.len, 1, 1))
        return self.main(input)


netg = NetG(128)
input = Variable(torch.randn(2, 4096))
out = netg(input)
print(out)
