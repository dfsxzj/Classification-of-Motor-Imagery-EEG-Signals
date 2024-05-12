import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(3*in_features, out_features))
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        input = torch.unsqueeze(input, dim=1)
        support = torch.matmul(adj, input)
        support = support.transpose(1,2)
        support = support.reshape(support.shape[0]*support.shape[1], support.shape[2]*support.shape[3])
        output = torch.matmul(support, self.weight)
        output = output.reshape(input.shape[0], input.shape[2], output.shape[1])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
