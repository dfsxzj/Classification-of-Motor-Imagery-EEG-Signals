import random
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch
import h5py
import os

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data():
    www = h5py.File(os.path.join(os.path.dirname(os.getcwd()),'脑电信号处理/feature.mat'))
    W = www['W'][:]
    # ccc = www['tz'][:25]
    ccc1 = www['tz'][:15]
    # ccc2=www['tz'][19:25]
    ccc2 = www['tz'][20:25]
    # ccc3=www['tz'][8:12]
    # ccc4=www['tz'][12:16]
    # ccc5=www['tz'][16:20]
    # ccc6=www['tz'][21:25]
    ccc = np.concatenate([ccc1, ccc2], axis=0)
    # ccc = www['data'][:]
    L = www['L'][:, :, :, :]
    labels = www['lab'][:]
    W = np.transpose(W, (2, 1, 0))
    ccc = np.transpose(ccc, (2, 1, 0))
    L = np.transpose(L, (3, 2, 1, 0))
    labels = np.transpose(labels, (1, 0))
    W = W.astype(np.float32)
    ccc = ccc.astype(np.float32)
    L = L.astype(np.float32)
    labels = torch.LongTensor(labels)
    W = torch.from_numpy(W)
    ccc = torch.from_numpy(ccc)
    L = torch.from_numpy(L)
    # labels = torch.from_numpy(labels)
    labels = torch.squeeze(labels) - 1

    # idx_train = random.sample(range(1800),1600)
    # idx_test = []
    # for i in range(1800):
    #     if i not in idx_train:
    #         idx_test.append(i)
    # num=np.array(range(1800))
    # i=0
    # idx_val = np.array(range(0+180*i,180+180*i))
    # idx_test = np.array(range(0+180*i,180+180*i))
    # idx_train = np.delete(num,idx_test)

    # idx_train = range(0,11000)
    # idx_test = range(11000,12800)
    return W, ccc, L, labels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    # label = torch.topk(labels, 1)[1].squeeze(1)
    preds = output.max(1)[1].type_as(labels)
    # pred = output.max(2)[1].type_as(labels)
    # preds = torch.mode(pred,dim=1)[0]
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def guiyihua(input):
    input = input.squeeze()
    mi, l = input.min(dim=1)
    mi = mi.unsqueeze(dim=1)
    temp = input - mi
    temps = temp.sum(dim=1)
    temps = temps.unsqueeze(dim=1)
    re = temp / temps
    re = re.unsqueeze(dim=2)
    return re


def jieduan(x, alpha, n):
    bb, cc = torch.topk(alpha, n, dim=1)  # 选取注意力向量大的前n个
    bb = bb[:, -1, :]
    bb = bb.repeat(1, 32)
    alpha = alpha.squeeze()
    pd = torch.ge(alpha, bb)
    alpha = pd * alpha
    alpha = guiyihua(alpha)
    # alpha = alpha.unsqueeze(dim=2)
    alpha = alpha.repeat(1, 1, x.shape[2])  # 最终的注意力向量
    x = x * alpha  # 最终特征按照注意力向量聚合
    return x


def pinjie(temp1, temp2):
    temp = temp1.repeat(1, 4)
    x = torch.cat((temp, temp2), 1)
    return x


def loss(output, labels):
    n = output.shape[0]
    ls = 0
    label = labels.unsqueeze(dim=1)
    label = label.repeat(1, 32)
    for i in range(n):
        ls = ls + F.nll_loss(output[i], label[i])
    ls = ls / n
    return ls


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
