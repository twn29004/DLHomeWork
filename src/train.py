import argparse

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

from MyDataset import MyDataset
from Modules import *

GAMA = 1    # 结构性损失的参数
LAMBDA = 10   # GP的参数
THETA = 1       # 特征匹配的参数

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="HW", help="dataset of train")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--n_Depochs", type=int, default=5, help="number of epochs of training Discriminator")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=30, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

torch.cuda.set_device(1)
cuda = True if torch.cuda.is_available() else False


# 准备数据
if opt.dataset == "HW":
    dataset = MyDataset("train.txt")
    opt.n_classes= 30
if opt.dataset == 'MNIST':
    dataset = dset.MNIST(root="../data", download=True,
                         transform=transforms.Compose([
                             transforms.Resize((32, 32)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ]))
    opt.n_classes = 10

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)


# 初始化模型及其参数
generator = Generator(opt.n_classes, opt.latent_dim)
discriminator = Discriminator(opt.channels, opt.n_classes)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 损失函数
# 第一个是对于预测真假的损失函数
adversarial_loss = torch.nn.BCELoss()
# 第二个是对于预测类别的损失函数
auxiliary_loss = torch.nn.CrossEntropyLoss()

# 设置优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 使用GPU加速
if cuda:
    print("cuda is available！！！now device id is ", torch.cuda.current_device())
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()


# 训练模型
for epoch in range(opt.n_epochs):
    d_loss = None
    g_loss = None
    for j, (imgs, labels) in enumerate(dataloader, 0):
        batch_size = imgs.shape[0]
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        optimizer_D.zero_grad()
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # 生成噪音和随机标签
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # 根据噪音和随机标签生成假图像
        gen_imgs = generator(z, gen_labels)
        # 将真假图片输入判别器并获得相应得分数
        real_pred, real_aux, real_conv = discriminator(real_imgs)
        fake_pred, fake_aux, fake_conv = discriminator(gen_imgs.detach())
        # 使得真假得值向应属于他们得值靠近，即真图片向1靠近，假图片像0靠近
        loss_cgan = adversarial_loss(real_pred, valid) + adversarial_loss(fake_pred, fake)

        # 注意这里label讲道理只应让真图片得预测标签项真标签靠近就好了，不需要假图片的标签也向其标签靠近吧
        # 另一种解释是不管真假，判别器应该让标签尽可能的对
        loss_label = auxiliary_loss(real_aux, labels) + auxiliary_loss(fake_aux, gen_labels)

        # 这个值越小，说明真假数分布的越像，所以在判别器中，要区分真假数据应尽可能使他们大
        # 但是这个真假数据并不是同一个标签的，感觉这种不是很有意义的样子
        loss_ofm = 0
        for index in range(len(real_conv)):
            loss_ofm += (real_conv[index] - fake_conv[index]).norm(1)
        # 这个的目的是让这项损失小一点，不然他在损失项中的比例太大了
        loss_ofm /= (32 ** 3)

        # 梯度损失，在原论文的代码中，这项是加的，意思就是这项越小越好
        GP = calc_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)

        # 计算判别器的损失，尽可能的分别出真假数据。
        # print("loss_cgan = {}, loss_label = {}, GP = {}".format(loss_cgan.item(), loss_label.item(), GP))
        loss_d = loss_cgan + loss_label + LAMBDA * GP
        loss_d.backward()
        d_loss = loss_d.item()
        optimizer_D.step()
        # 每5次训练一下生成器
        if j % 5 == 0:
            optimizer_G.zero_grad()
            # 根据真实标签生成图片，让生成的图片的数据分布尽可能向真实数据靠，这样ofm才有意义的柑橘
            gen_imgs = generator(z, labels)
            fake_pred, fake_aux, fake_conv = discriminator(gen_imgs)
            real_pred, real_aux, real_conv = discriminator(real_imgs)

            # 需要的是缩小判别器对生成的假数据与真实值之间的差距
            loss_cgan = adversarial_loss(fake_pred, valid)
            # 生成的加图片的预测的标签尽可能的向真图片靠
            loss_label = auxiliary_loss(fake_aux, gen_labels)
            loss_ofm = 0
            for index in range(len(real_conv)):
                loss_ofm += (real_conv[index] - fake_conv[index]).norm(1)
            # loss表示的是生成器欺骗判别器的能力
            loss_ofm /= (32 ** 3)
            # 图片的一致性损失
            loss_consis = (real_imgs - gen_imgs).norm(1) / (32 ** 4)
            # 这个损失减小，说明对生成的假数据与生成的标签
            # print("loss_cgan = {}, loss_label = {}, loss_consis = {}, loss_ofm = {}"
            #       .format(loss_cgan.item(), loss_label.item(), loss_consis.item(), loss_ofm.item()))
            loss_g = loss_cgan + loss_label + GAMA * loss_consis + THETA * loss_ofm
            loss_g.backward()
            g_loss = loss_g.item()
            optimizer_G.step()
    with open("train_mnist.txt", 'wt') as f:
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, d_loss, g_loss), file=f
        )
    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, d_loss, g_loss)
    )
    sample_image(generator, opt.latent_dim, opt.n_classes, epoch)
