import argparse

import numpy as np
import torch
import torch.nn as nn
from MyDataset import MyDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

GAMA = 10
LAMBDA = 10

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
parser.add_argument("--n_Depochs", type=int, default=5, help="number of epochs of training Discriminator")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=30, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 这个latent_dim指得视噪音的大小,这里生成了一个30 x 128的矩阵，给定一个[0：30]的数，返回一个固定的噪音
        # 换句话说就是将噪音和标签绑定
        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 256 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            # 256 x 4 x 4 -> 256 x 8 x8
            nn.Upsample(scale_factor=2),
            # 256 x 8 x8 -> 128 x 8 x 8
            nn.Conv2d(256, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8 -> 128 x 16 x 16
            nn.Upsample(scale_factor=2),
            # 128 x 16 x 16 -> 64 x 16 x 16
            nn.Conv2d(128, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 16 x 16 -> 64 x 32 x 32
            nn.Upsample(scale_factor=2),
            # 64 x 32 x 32 -> 1 x 32 x 32
            nn.Conv2d(64, opt.channels, 5, stride=1, padding=2),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        # 进行FC
        out = self.l1(gen_input)
        # FC之后在reshape
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 5, 2, 2),
                nn.LeakyReLU(0.2, inplace=True))
            if bn:
                block.add_module(module=nn.BatchNorm2d(out_filters, 0.8), name="bn")
            return block

        self.conv1 = discriminator_block(opt.channels, 64, bn=False)
        self.conv2 = discriminator_block(64, 128)
        self.conv3 = discriminator_block(128, 256)

        # The height and width of down sampled image
        ds_size = 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, opt.n_classes), nn.Softmax(dim=1))

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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 用于计算判别器中的梯度损失项
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batch_size(), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                        disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Loss functions
# 第一个是对于预测真假的损失函数
adversarial_loss = torch.nn.BCELoss()
# 第二个是对于预测类别的损失函数
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    print("cuda is available！！！")
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

dataset = MyDataset("train.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
dataloaderD = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    # 先训练n_Depoches判别器
    d_loss = None
    g_loss = None
    for i in range(opt.n_Depochs):
        print(i)
        for j, (imgs, labels) in enumerate(dataloaderD, 0):
            batch_size = imgs.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            # 准备好数据之后开始训练
            optimizer_D.zero_grad()
            real_pred, real_aux, real_conv = discriminator(real_imgs)
            fake_pred, fake_aux, fake_conv = discriminator(gen_imgs.detach())
            loss_cgan = adversarial_loss(real_pred, valid) - adversarial_loss(fake_pred, fake)
            loss_label = auxiliary_loss(real_aux, labels) + auxiliary_loss(fake_aux, gen_labels)
            loss_ofm = 0
            for index in range(len(real_conv)):
                loss_ofm += (real_conv[index] - fake_conv[index]).norm(1)
            # GP = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
            GP = 1

            # 计算判别器的损失
            loss_d = loss_cgan - loss_label - GAMA * loss_ofm - LAMBDA * GP
            loss_d.backward()
            d_loss = loss_d.item()
            optimizer_D.step()
        print("the discriminator epoch {}, and the loss of discriminator is {}".format(j, d_loss))

    # 训练生成器
    for k, (imgs, labels) in enumerate(dataloader, 0):
        batch_size = imgs.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        # real data
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        # 开始训练
        discriminator.zero_grad()
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        real_pred, real_aux, real_conv = discriminator(real_imgs)
        fake_pred, fake_aux, fake_conv = discriminator(gen_imgs.detach())
        loss_cgan = adversarial_loss(real_pred, valid) - adversarial_loss(fake_pred, fake)
        loss_label = auxiliary_loss(real_aux, labels) + auxiliary_loss(fake_aux, gen_labels)
        loss_ofm = 0
        for index in range(len(real_conv)):
            loss_ofm += (real_conv[index] - fake_conv[index]).norm(1)
        loss_consis = (real_imgs - gen_imgs).norm(1)
        loss_g = loss_cgan + loss_label + GAMA * loss_ofm + LAMBDA * loss_consis
        loss_g.backward()
        g_loss = loss_g.item()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, d_loss, g_loss)
        )
        batches_done = epoch * len(dataloader) + k
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
