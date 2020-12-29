import argparse

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image


torch.cuda.set_device(1)
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(Generator, self).__init__()
        # 这个latent_dim指得视噪音的大小,这里生成了一个30 x 64的矩阵，给定一个[0：30]的数，返回一个固定的噪音
        # 换句话说就是将噪音和标签绑定
        self.init_size = 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + n_classes, 256 * self.init_size ** 2))
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            # 256 x 4 x 4 -> 128 x 8 x 8
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
            # 128 x 8 x 8 -> 64 x 16 x 16
            nn.ConvTranspose2d(128, 64, 5, 2, 2, 1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
            # 64 x 16 x 16 -> 1 x 32 x 32
            nn.ConvTranspose2d(64, 1, 5, 2, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # 将label转化成one-hot
        tmp = torch.zeros((labels.size(0), self.n_classes)).cuda()
        for index in range(0, labels.size(0)):
            tmp[index][labels[index]] = 1
        # noise与tmp进行拼接
        gen_input = torch.cat((tmp, noise), dim=1)
        # 进行FC
        out = self.l1(gen_input)
        # FC之后在reshape
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_channels, n_classes):
        super(Discriminator, self).__init__()
        self.channels = img_channels
        self.n_classes = n_classes
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 5, 2, 2),
                nn.LeakyReLU(0.2, inplace=True))
            if bn:
                block.add_module(module=nn.BatchNorm2d(out_filters, 0.8), name="bn")
            return block

        self.conv1 = discriminator_block(self.channels, 64, bn=False)
        self.conv2 = discriminator_block(64, 128)
        self.conv3 = discriminator_block(128, 256)

        # The height and width of down sampled image
        ds_size = 4
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, self.n_classes), nn.Softmax(dim=1))

    def forward(self, img):
        # 用于特征匹配计算损失
        outConv1 = self.conv1(img)
        outConv2 = self.conv2(outConv1)
        outConv3 = self.conv3(outConv2)
        out = outConv3
        # 经过conv_locks之后将输出结果转化为一维向量
        out = out.view(out.shape[0], -1)
        # 对这个一维向量分别输入两个线性层，一个是预测真假，一个是预测属于某个类别的概率
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label, [outConv1, outConv2, outConv3]


# 用于初始化网络的参数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 用于计算判别器中的梯度损失项
def calc_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _, _ = D(interpolates)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# 生成可视化图片
def sample_image(G, latent_dim, n_row, batches_done):
    # 生成噪音
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # 生成每一个类的标签
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = G(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
    print(batches_done, " image has been save")