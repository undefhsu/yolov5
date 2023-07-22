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
    with torch.no_grad():
        count = 1
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            count += 1
            print("count:", count)
            output = model(data)

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
        label = int(image_index[
                        -5])  # 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。我这里是"E:\\Python Project\\Pytorch\\dogs-vs-cats\\train\\cat.0.jpg"，所以先用"\\"分割，选取最后一个为['cat.0.jpg']，然后使用"."分割，选取[cat]作为该图片的标签
        print("Image:", img_path, label)
        sample = (img, label)  # 根据图片和标签创建字典

        if self.transform:
            sample = self.transform(img)  # 对样本进行变换
        return sample, label  # 返回该样本
