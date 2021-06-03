import torch

x_data = [1.0, 2.0, 3.0]  # 构建数据集
y_data = [2.0, 4.0, 6.0]
w = torch.Tensor([1.0])  # w是权重，创建tensor变量，假设现在w只有一个值，就用1.0，要用中括号
w.requires_grad = True  # 表示需要计算梯度


# sum=0#总的loss
def forward(x):  # 模型
    return x * w  # 注意w是一个tensor,所以这里的*已经被重载，表示tensor与tensor之间的数乘


# 但是x不一定是一个tensor所以要进行自动类型转换，把x转换成一个Tensor

def loss(x, y):  # 损失函数,每调用一次loss函数就把计算图动态构建出来了
    y_pred = forward(x)  # y_pred=w*x
    return (y_pred - y) ** 2  # 平方
    # 训练过程
    print("predict(before training)", 4, forward(4).item())


for epoch in range(100):  # 训练100轮
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量  #构建计算图的过程都直接用张量
        l.backward()  # 把计算链路上的梯度都求出来
        # 梯度会存到w里面，之后计算图就会被释放
        # 下一次调用Loss会创建一个新的计算图
        # 为什么不把计算图保留呢：因为构建神经网络的时候，每一次运行的时候计算图可能是不一样的，
        # 所以每进行一次反向传播，就把图释放，准备下一次的图，这是一个非常灵活的方式，也是pytorch的核心竞争力
        # sum+=l;#这样是不合理的，l是一个张量，所以sum是关于l的一个计算图，sum里面下一次又多了一个指向l的引用
        # 经过反复的训练，计算图一致在内存中保存，当训练的轮数很多，计算图将会越做越大，就把内存吃光了
        # 所以计算的时候l一般用标量
        #  sum+=l.item()#用l.item()把损失值取出来，不要直接用l，l是tensor计算会构建计算图
        print('\tgrad:', x, y, w.grad.item())  # .item()是用来把梯度里面的数值直接拿出来，变成一个标量（也是为了防止产生计算图）
        # 权重更新
        w.data = w.data - 0.01 * w.grad.data  # 因为w.grad是一个张量，张量计算是会建立计算图的，所以我们先取到w.grad的data
        # w.grad的data计算是不会建立计算图的
        # 现在做的是对权重的数值做修改，不用对修改数值的过程求梯度，只是纯数值的修改，所以要用.dat
        w.grad.data.zero_()  # 把权重里面梯度的数据全都清零，权重更新之后导数是还存在的，如果不把导数清零，下次再计算出来的导数就会与上一次计算出来的导数相加
        # 就不是我们想要的了
        print("progress:", epoch, l.item())
        #  print("progress:",epoch,l.item())
print("predict(after training)", 4, forward(4).item())
