from __future__ import division
from __future__ import print_function

import random
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


import torch
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils import load_data, accuracy, loss
from model import GCN,CNN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=[16, 32, 64, 128, 256, 512],  # nhid models line29
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
W, features, adj, labels= load_data()
acc_cross=np.ones((10))*0
acc_cross_rf=np.ones((10))*0
acc_cross_svm=np.ones((10))*0
acc_cross_nn=np.ones((10))*0
acc_cross_knn=np.ones((10))*0
acc_cross_cnn=np.ones((10))*0
cors = int(W.shape[0]/10)

for i in range(10):
    num = np.array(range(W.shape[0]))
    idx_val = np.array(range(0 + cors * i, cors + cors * i))
    idx_test = np.array(range(0 + cors * i, cors + cors * i))
    idx_train = np.delete(num, idx_test)

    # other_models
    X_train_other = features[idx_train].clone().reshape(idx_train.shape[0],1280)
    y_train_other = labels[idx_train]
    X_test_other = features[idx_test].clone().reshape(idx_test.shape[0],1280)
    y_test_other = labels[idx_test]

    model_other = RandomForestClassifier()
    model_other.fit(X_train_other, y_train_other)
    acc_cross_rf[i] = model_other.score(X_test_other,y_test_other)

    model_other = SVC(kernel='rbf')
    model_other.fit(X_train_other, y_train_other)
    acc_cross_svm[i] = model_other.score(X_test_other,y_test_other)

    model_other = KNeighborsClassifier(n_neighbors=21)
    model_other.fit(X_train_other, y_train_other)
    acc_cross_knn[i] = model_other.score(X_test_other,y_test_other)

    model_other = torch.nn.Sequential(
        torch.nn.Linear(1280, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    optimizer = optim.Adam(model_other.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(600):
        optimizer.zero_grad()
        outputs = model_other(X_train_other)
        loss = F.nll_loss(F.log_softmax(outputs,dim=1), y_train_other)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model_other(X_test_other)
        _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test_other).sum().item()
    total = y_test_other.size(0)
    acc_cross_nn[i] = correct / total

    model_other = CNN(num_classes=4)
    model_other.cuda()
    optimizer = optim.Adam(model_other.parameters(), lr=0.001)
    features = features.cuda()
    labels = labels.cuda()
    X_train_other = features[idx_train].clone().reshape(idx_train.shape[0], 1280)
    y_train_other = labels[idx_train]
    X_test_other = features[idx_test].clone().reshape(idx_test.shape[0], 1280)
    y_test_other = labels[idx_test]
    for epoch in range(600):
        # 将输入数据传递给模型
        outputs = model_other(features[idx_train].view(idx_train.shape[0], 1, features.shape[1], features.shape[2]))
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), y_train_other)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model_other(features[idx_test].view(idx_test.shape[0], 1, features.shape[1], features.shape[2]))
        _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test_other).sum().item()
    total = y_test_other.size(0)
    acc_cross_cnn[i] = correct / total


###### GCNN #########
for i in range(10):
    num = np.array(range(W.shape[0]))
    idx_val = np.array(range(0 + cors * i, cors + cors * i))
    idx_test = np.array(range(0 + cors * i, cors + cors * i))
    idx_train = np.delete(num, idx_test)

    # Model and optimizer
    model = GCN(nfeat=features.shape[2],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    losstrain=[]
    acctrain=[]
    lossval=[]
    accval=[]
    def train(epoch):
        t = time.time()
        batchsize = 512
        bat = random.sample(list(idx_train), batchsize)
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        losstrain.append(loss_train.item())
        acc_train = accuracy(output[idx_train], labels[idx_train])
        acctrain.append(acc_train.item())
        loss_train.backward()
        optimizer.step()


        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        lossval.append(loss_val.item())
        accval.append(acc_val.item())
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        #loss_test = loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test

    # Train model

    t_total = time.time()
    for epoch in range(args.epochs//2):
        train(epoch)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
    for epoch in range(args.epochs//2,args.epochs):
        train(epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    acc_cross[i]=test().item()
plt.plot(acc_cross)
plt.show()
plt.plot(acc_cross, label = 'GCNN')
plt.plot(acc_cross_nn, label='Neural Network')
plt.plot(acc_cross_rf, label='Random Forest')
plt.plot(acc_cross_knn, label='KNN')
plt.plot(acc_cross_svm, label='SVM')
plt.plot(acc_cross_cnn, label='CNN')
plt.title('Comparison of Classification Algorithms')
plt.xlabel('cross')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print(acc_cross.mean())
