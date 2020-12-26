from __future__ import print_function

import torch.nn as nn


class Generator(nn.Module):
    # lenZ表示输入向量的长度, ngF表示最终生成器生成图片的长宽高
    def __init__(self, lenZ):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 反卷积层
            # 输入是一个128 + 30的
            nn.ConvTranspose2d(lenZ, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, 2, 0, bias=False)
        )


class Discriminator(nn.Module):
    def __int__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1x32x32 -> 64x16x16
            nn.Conv2d(1, 64, 5, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64x16x16 -> 128x8x8
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128x8x8 -> 256x4x4
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 做一个分类
            nn.Sigmoid()
        )
