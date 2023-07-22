from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch
import torchvision
import numpy as np
from torchvision import transforms


def test(model, device, test_loader):
    '''
    dataset the model
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output.shape)
            # TODO： 验证结果，与label进行比对
            # 问题在于 output是个Tensor list, tensor数量为batch-size， 而target永远是一个Tensor

            # pred.shape有四个维度，而target.shape只有一个，因此尝试将pred只保留一个维度的信息
            pred = output[0].argmax(dim=1, keepdim=True).argmax(dim=2, keepdim=True).argmax(dim=3, keepdim=True)
            print(pred.shape, target.shape)
            print(pred, target.view_as(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), acc))

    return


class MyDataset(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = Image.open(img_path).convert('RGB')  # 读取该图片, 维度为3通道的RGB图片
        # TODO: 找label，anchor之类的东西
        label = int(image_index[-5])  # 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。我这里是"E:\\Python Project\\Pytorch\\dogs-vs-cats\\train\\cat.0.jpg"，所以先用"\\"分割，选取最后一个为['cat.0.jpg']，然后使用"."分割，选取[cat]作为该图片的标签
        sample = (img, label)  # 根据图片和标签创建字典

        if self.transform:
            sample = self.transform(img)  # 对样本进行变换
        return sample, label  # 返回该样本
