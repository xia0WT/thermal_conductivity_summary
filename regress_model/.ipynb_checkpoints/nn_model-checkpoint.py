import torch
from torch import nn
from torch.nn import functional as F
from regress_model.nn_base_model import Highway


# 定义我们自己的模型类，需要继承自 torch.nn.Module。
# 这样方便我们直接使用pytorch实现的各种机器学习底层方法。
class NNModel(nn.Module):

    # 类的构造方法，在后续运行NNModel(13)的时候，就会自动调用这一方法
    # self指的是实例本身, num_features是我们定义的传入参数：
    # 例如当我们运行 model = NNModel(13)，13就是num_features，self 就是model自身
    def __init__(self, num_features):
        # 下面这一行是运行父类(nn.Module)的构造方法
        super(NNModel, self).__init__()

        # 定义多个全连接层
        self.l1 = nn.Linear(num_features, 1024)
        self.h1 = Highway(1024)
        self.h2 = Highway(1024)
        self.h3 = Highway(1024)
        self.h4 = Highway(1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 1)
        # 注意：任意数据输入sigmoid，输出的值都会处于0到1之间
        self.sigmoid = nn.Sigmoid()

    # 后续我们使用 model = NNModel(13)实例化了一个模型之后，
    # 使用 pred = model(features)即可获得预测值，
    # 其中 model(features) 时，就是在运行下方的forward方法
    def forward(self, features):
        x = F.relu(self.l1(features))
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # x = self.l1(features)
        x = F.relu(x)
        # x = self.l3(x)
        y = self.sigmoid(self.l4(x))
        # 注意最后返回的预测值是sigmoid的输出，一定会处于0到1之间
        # 正好符合归一化之后的房价的范围
        return y
