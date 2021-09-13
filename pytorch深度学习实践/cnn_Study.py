# -*- coding: utf-8 -*- 
# @Time : 2021/6/1 14:37 
# @Author : Lee 
# @File : cnn_Study.py
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

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
①前向传播：卷积--池化--激活--全连接（分类）  ②反向传播
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

"""
1.nn.CrossEntropyLoss()函数计算交叉熵损失;
  注意，使用nn.CrossEntropyLoss()时，不需要现将输出经过softmax层，否则计算的损失会有误，即直接将网络输出用来计算损失即可.
2.model.parameters()是获取model网络的参数，构建好神经网络后，网络的参数都保存在parameters()函数当中。
3.lr是学习率
4.momentum
5.SGD 随机梯度下降
"""
# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# training cycle forward, backward, update

"""
1.enumerate()
①如果对一个列表，既要遍历索引又要遍历元素时
list1 = ["这", "是", "一个", "测试"]
for index, item in enumerate(list1):
    print index, item
>>>
0 这
1 是
2 一个
3 测试
②enumerate可以接收第二个参数，用于指定索引起始值，如：
list1 = ["这", "是", "一个", "测试"]
for index, item in enumerate(list1, 1):
    print index, item
>>>
1 这
2 是
3 一个
4 测试


2.optimizer.zero_grad()
由于pytorch的动态计算图，当我们使用loss.backward()和optimizer.step()进行梯度下降更新参数的时候，梯度并不会自动清零。并且这两个操作是独立操作。
进行backward之前需要清零梯度。

3.optimizer.step():
  只有用了optimizer.step()，模型参数才会更新
"""


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()  # 降梯度归零

        outputs = model(inputs)
        loss = criterion(outputs, target)  # 计算损失函数
        loss.backward()  # 反向传播计算得到每个参数的梯度值
        optimizer.step()  # 通过梯度下降执行一步参数更新（optimizer.step()）。一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数。

        running_loss += loss.item()  # 计算一个epoch的损失，因为累加了loss。
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


"""
1.torch.no_grad():
  不需要计算梯度，因为是测试，只需要输入数据用模型预测。训练时需要反向计算梯度，然后更新参数、模型，

2.torch.max():
  这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index（也就是predicted）。
  
3.为什么这里选择用下划线?
  这是因为我们不关心最大值是什么，而关心最大值对应的index是什么，所以选用下划线代表不需要用到的变量。当然用字母也行。
  
4.dim=1表示输出所在行的最大值，若改写成dim=0则输出所在列的最大值。
"""


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)  # total为测试的个数，correct为测试正确的个数；
            correct += (predicted == labels).sum().item()
    print('在测试集上的正确率: %.3f %% ' % (100 * correct / total))
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
    plt.ylabel('正确率')
    plt.xlabel('epoch')
    plt.show()
