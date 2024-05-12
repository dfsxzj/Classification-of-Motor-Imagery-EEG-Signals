import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid[0])
        self.bn1 = nn.BatchNorm1d(nhid[0],affine=True)

        self.gc2 = GraphConvolution(nhid[0], nhid[1])
        self.bn2 = nn.BatchNorm1d(nhid[1],affine=True)

        self.pool = nn.MaxPool1d(2, stride=2)


        self.fc3 = nn.Sequential(
            nn.Linear(64 * nhid[1], 16 * nhid[1]),
            nn.ReLU(),
            nn.Linear(16 * nhid[1], 4 * nhid[1]),
            nn.ReLU(),
            nn.Linear(4 * nhid[1], 4),
        )
        self.dropout = dropout


    def forward(self, x, adj):
        # 第一块
        x = self.gc1(x,adj)
        x = torch.transpose(x,2,1)
        x = self.bn1(x)
        x = torch.transpose(x,2,1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x,adj)
        x = torch.transpose(x,2,1)
        x = self.bn2(x)
        x = torch.transpose(x,2,1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(64 * 13 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 64 * 13 * 3)  # flatten操作
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x