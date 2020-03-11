# 定义CoraData类对数据进行预处理，主要包括数据下载、规范数据并进行缓存以备重复使用
# 最终得到的数据形式包括如下几部分：
# x:节点特征，维度为2808X1433
# y:节点对应的标签，包括7个类别
# adjacency：邻接矩阵，维度为2708X2708
# train_mask、val_mask、test_mask：与节点数相同的掩码，用于划分训练集、验证集、测试集

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple
import os
import os.path as osp
import pickle
import urllib
import numpy as np
import itertools


# 用于保存处理好的数据
# namedtuple是继承自tuple的子类。namedtuple创建一个和tuple类似的对象，而且对象拥有可访问的属性。
Data = namedtuple('data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])

class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora.{}".format(name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
    def __init__(self, data_root="cora", rebuild=False):
        # 包括数据集下载、处理、加载等功能
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()
            self._data = self.process_data()
        with open(save_file, 'wb') as f:
            pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))
    @property
    def data(self):
        return self.data

    def maybe_download(self):
        save_path=os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data("{}/ind.cora{}".format(self.download_url, name), save_path)
    @staticmethod
    def download_data(url, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            data = urllib.request.urlopen(url)
            filename = os.path.splitext(url)
            with open(os.path.join(save_path, filename), 'wb') as f:
                f.write(data.read())
                return True

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, "raw", name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())
        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""

        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
        edge_index.extend([v, src] for v in dst)
        # 由于上述得到的结果中存在重复的边，删掉这些重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                       (edge_index[:, 0], edge_index[:, 1])),
                                      shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""

        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

# 根据GCN的定义来定义GCN层，代码直接根据定义来实现，需要特别注意的是邻接矩阵是稀疏矩阵，为了提高运算效率，使用了稀疏矩阵的乘法。如代码清单5 - 3
# 所示：
# 代码清单5 - 3　GCN层定义
class GraphConvolution(nn.Module):
    # 图卷积：L*X*\theta
    #     # input-dim:int  节点输入特征的维度
    #     # output_dim:ini 输出特征维度
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output
#
# 有了数据和GCN层，就可以构建模型进行训练了。定义一个两层的GCN，其中输入的维度为1433，隐藏层维度设为16，最后一层GCN将输出维度变为类别数7，激活函数使用的是ReLU。
# 两层GCN的模型

class GcnNet(nn.Module):
    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits

def normalization(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5"""
    adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()

# 超参数定义
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200
# 模型定义，包括模型实例化、损失函数与优化器定义
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GcnNet().to(device)
# 损失函数使用交叉熵
criterion = nn.CrossEntropyLoss().to(device)
# 优化器使用Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 加载数据，并转换为torch.Tensor
dataset = CoraData().data
x = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalize_adjacency = normalization(dataset.adjacency)  # 规范化邻接矩阵
indices = torch.from_numpy(np.asarray([normalize_adjacency.row,
                                                   normalize_adjacency.col])).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values,
                                                        (2708, 2708)).to(device)

#
# 所有准备工作都做好后，就可以根据神经网络运行流程进行模型训练了
# 所示，通过不断迭代优化，我们将记录训练过程中损失值的变化和验证集上的准确率，训练完成后在测试集上测试模型的效果。
# 模型训练与测试


def train():
    print("test=========")
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)  # 前向传播
    train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督
    loss = criterion(train_mask_logits, train_y)  # 计算损失值
    optimizer.zero_grad()
    loss.backward()  # 反向传播计算参数的梯度
    optimizer.step()  # 使用优化方法进行梯度更新
    train_acc = test(tensor_train_mask)  # 计算当前模型在训练集上的准确率
    val_acc = test(tensor_val_mask)  # 计算当前模型在验证集上的准确率
    # 记录训练过程中损失值和准确率的变化，用于画图
    loss_history.append(loss.item())
    val_acc_history.append(val_acc.item())
    print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
    epoch, loss.item(), train_acc.item(), val_acc.item()))
    return loss_history, val_acc_history

def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
    test_mask_logits = logits[mask]
    predict_y = test_mask_logits.max(1)[1]
    accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy


train()