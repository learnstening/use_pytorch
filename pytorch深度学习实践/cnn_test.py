# -*- coding: utf-8 -*- 
# @Time : 2021/6/1 14:37 
# @Author : Lee 
# @File : cnn_test.py
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# prepare dataset
"""
1.torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起。
2.transforms.ToTensor()：
    （1）transforms.ToTensor() 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之间
    （2）通道的具体顺序与cv2读的还是PIL.Image读的图片有关系
        cv2:(B,G,R)
        PIL.Image:(R, G, B)
3.transforms.Normalize(mean,std)：标准化。
4.torch.utils.data.DataLoader：功能：构建可迭代的数据装载器：
    dataset:Dataset类，决定数据从哪里读取及如何读取；
    batchsize：批大小；
    num_works:是否多进程读取数据；
    shuffle：每个epoch是否乱序；
    drop_last：当样本数不能被batchsize整除时，是否舍弃最后一批数据；
    Epoch：所有训练样本都已输入到模型中，称为一个Epoch；
    
5.
    Iteration：一批样本输入到模型中（所有的样本需要好几次才能完全送到模型里），称之为一个Iteration；
    Batchsize：批大小(一批有几个样本)，决定一个Epoch中有多少个Iteration；
        
    样本总数：80，Batchsize：8 （样本能被Batchsize整除）
    1 Epoch = 10 Iteration
        
    样本总数：87，Batchsize=8 （样本不能被Batchsize整除）
    1 Epoch = 10 Iteration，drop_last = True
    1 Epoch = 11 Iteration， drop_last = False
"""

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='./dataset/mnist/', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# design model using class

"""
卷积神经网络运作流程：
①前向传播：卷积--池化--激活--全连接  ②反向传播
1.Conv2d:二维卷积
2.ReLu:对于输入的负值，输出全为0，对于正值，原样输出。
"""


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)  # 卷积运算
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)  # 池化层
        self.fc = torch.nn.Linear(320, 10)  # 全连接层

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)

        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是320
        # print("x.shape",x.shape)
        x = self.fc(x)

        return x


model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %.3f %% ' % (100 * correct / total))
    return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
