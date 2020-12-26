# import torch
# from torch import nn
# from torch.autograd import Variable
#
#
# class NetG(nn.Module):
#     def __init__(self, lenZ):
#         super(NetG, self).__init__()
#         self.len = lenZ
#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 5, 2, 2, bias=False)
#         )
#
#     def forward(self, input):
#         return self.main(input)
#
#
# # netg = NetG(128)
# input = Variable(torch.randn(2, 2, 4, 4))
# print(input.norm(1))
# # out = netg(input)
# # print(out.size())

import random

tmp = random.sample(range(0, 20), 10)
print(len(tmp))
