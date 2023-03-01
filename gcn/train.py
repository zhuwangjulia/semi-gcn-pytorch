from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
#画图可视化损失
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from d2l import torch as d2l
import torch
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
path1 = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path1)
from gcn.utils import load_data, accuracy
from gcn.models import GCN
from gcn.animator import Animator

#print({path1})

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#print({args.cuda})

# Empty GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden, #使用了前面的随机种子
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
#优化方法选择adam
#优化方法还有SGD BGD
#optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay= args.weight_decay) #手动调整学习率，而非衰减

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


'''if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    '''

# 训练过程
def train(epoch):    
    t = time.time()
    model.train() 
    optimizer.zero_grad()
    output = model(features, adj) #为什么model变成函数
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    #计算验证集
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    #eval_losses += loss_val.item()
    #eval_acces += acc_val.item()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t), file = mylog)
    
    return loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()
    


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])  
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()), file=mylog)
    #return loss_test, acc_test

# Train model
t_total = time.time()
# 将 train accuracy 保存到 "tensorboard/train" 文件夹
log_dir = os.path.join('tensorboard', 'train')
train_writer = SummaryWriter(log_dir=log_dir)
train_loss_writer = SummaryWriter(log_dir=log_dir)
# 将 test accuracy 保存到 "tensorboard/validation" 文件夹
log_dir = os.path.join('tensorboard', 'validation')
val_writer = SummaryWriter(log_dir=log_dir)
val_loss_writer = SummaryWriter(log_dir=log_dir)
    
#定义loss, acc字符串   
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

'''animator = Animator(xlabel='epoch', xlim=[1, args.epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'val loss', 'val acc'])
                        '''
#迭代训练
mylog = open('output.txt', mode = 'w',encoding='utf-8')
for epoch in range(args.epochs):
    out = train(epoch)
    #动画显示
    #figure = animator.add(epoch + 1, out)
    # 写入文件
    train_writer.add_scalar('Accuracy', out[1], epoch)
    train_loss_writer.add_scalar('Loss', out[0],epoch)
    val_writer.add_scalar('Accuracy', out[3], epoch)
    val_loss_writer.add_scalar('Loss',out[2],epoch)
    
    train_loss_list.append(out[0])
    train_acc_list.append(out[1])
    val_loss_list.append(out[2])
    val_acc_list.append(out[3])
print("Optimization Finished!", file=mylog)
print("Total time elapsed: {:.4f}s".format(time.time() - t_total), file=mylog)

#plt绘图代码
markers = {'<!-- --> train': 'o', 'val': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, val_acc_list, label='val acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
title = plt.title('Accuracy')
plt.savefig('acc.png')
#plt.show()
print('acc complete')

marker_loss = {'<!-- --> train': 'o', 'val': 's'}
y = np.arange(len(train_loss_list))
plt.plot(y, train_loss_list, label='train loss')
plt.plot(y, val_loss_list, label='val loss', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 10)
plt.legend(loc='lower right')
title = plt.title('Loss')
plt.savefig('loss.png')
#plt.show()
print('loss complete')
# Testing
test_out = test()
print("Test Finished!", file=mylog)
mylog.close()