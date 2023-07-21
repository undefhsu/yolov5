'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Common functions for simple PyTorch MNIST example
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# 导入相关模块
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 10, kernel_size=3, stride=3),
            nn.BatchNorm2d(10),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.network(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    '''
    train the model
    '''
    model.train()
    counter = 0
    print("Epoch " + str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        x = model(data)
        output = F.log_softmax(input=x, dim=0)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        counter += 1


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
            pred = output.argmax(dim=1, keepdim=True)
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
        img = Image.open(img_path)  # 读取该图片
        label = int(image_index[
                        -5])  # 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。我这里是"E:\\Python Project\\Pytorch\\dogs-vs-cats\\train\\cat.0.jpg"，所以先用"\\"分割，选取最后一个为['cat.0.jpg']，然后使用"."分割，选取[cat]作为该图片的标签
        sample = (img, label)  # 根据图片和标签创建字典

        if self.transform:
            sample = self.transform(img)  # 对样本进行变换
        return sample, label  # 返回该样本
