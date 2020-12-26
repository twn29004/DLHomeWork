import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# 数据集位置
# 注意这里/data是因为使用的是linux
root = os.path.abspath(os.path.join(os.getcwd(), "../data"))


# 读取文件的格式
def default_loader(path):
    if not os.path.exists(path):
        print(path + " is not exist!!!")
        exit(-1)
    return Image.open(path).convert('L')


default_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])


class MyDataset(Dataset):
    # txtFile就是labletxt文件名
    def __init__(self, txtFile, transform=default_transform, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        txtFile = root + '/' + txtFile
        if not os.path.isfile(txtFile):
            print(txtFile + " is not exist!!!")
            exit(-1)
        print(txtFile + "has been loaded.")
        labelFile = open(txtFile, 'r')
        imgs = []
        cnt = 0
        self.label2index = dict()
        for line in labelFile:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
            index = int(words[1])
            if index not in self.label2index.keys():
                self.label2index[index] = cnt
                cnt += 1
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(root + fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label2index[int(label)]

    def __len__(self):
        return len(self.imgs)
