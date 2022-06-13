import numpy

import dataset
from dataloader import LIBSVMLoader
import numpy as np
from sklearn.model_selection import KFold
import random
import math
#import wandb
import neural_network

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F

#wandb.init(project="my-test-project", entity="ziwencheng")

def get_model(lr,input_dim, hidden_dim, output_dim):
    model = neural_network.Neural_Network(input_dim, hidden_dim, output_dim)
    return model, optim.SGD(model.parameters(), lr=lr)

def loss_batch(model, loss_func, xb, yb, opt=None):
    pred = model(xb.float())
    loss = loss_func(pred, yb)
    pred_indices = torch.argmax(pred, dim=1)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), pred_indices

def fit(model, loss_func, opt, train_dl, test_dl):
    #for epoch in range(epochs):
        train_loss, train_correct = 0.0, 0
        model.train()
        for xb, yb in train_dl:
            loss, pred_indices = loss_batch(model, loss_func, xb, yb, opt)
            train_loss += loss * len(xb)
            yb_indices = torch.argmax(yb, dim=1)
            train_correct += (pred_indices == yb_indices).sum().item()
            """"
            correct_torch = torch.eq(pred_indices, yb_indices)
            correct_numpy = 1 * correct_torch.numpy()
            train_correct = correct_numpy.sum()
            """
        valid_loss, val_correct = 0.0, 0
        model.eval()
        with torch.no_grad():
            for xb, yb in test_dl:
                val_loss, valpred_indices = loss_batch(model, loss_func, xb, yb)
                valid_loss += val_loss * len(xb)
                valyb_indices = torch.argmax(yb, dim=1)
                val_correct += (valpred_indices == valyb_indices).sum().item()

        return train_loss, train_correct, valid_loss, val_correct


def getTensorDataset(name):
    x, y = dataset.get_precise_data(name)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    name_ds = TensorDataset(xt, yt)
    return name_ds

def getTensorImpreciseDataset(name, corruption):
    x, y = dataset.get_imprecise_data(name, corruption)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    name_ds = TensorDataset(xt, yt)
    return name_ds
"""
k = 10
batch_size = 8
lr = 0.2
x, y = map(torch.tensor, dataset.get_precise_data("svmguide2"))
svmguide2_ds = TensorDataset(x, y)
epochs = 10
ce_loss_func = F.cross_entropy
#train_dl = DataLoader(svmguide2_ds, batch_size=batch_size)
#foldperf={}
input_dim = 20
hidden_dim = 5
output_dim = 3

wandb.config = {
  "learning_rate": 0.1,
  "epochs": 2,
  "batch_size": 8
}
"""
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def OSL_CrossEntropy(pred, yb):
    loss_sum = torch.Tensor([0.0])
    for i in range(len(yb)):
        # if torch.count_nonzero(yb[i]) >= 1:
        target_indices = (yb[i] == 1).nonzero(as_tuple=True)[0]
        result = torch.take(log_softmax(pred[i]), target_indices)
        loss = torch.max(result).multiply(-1)
        loss_sum = loss_sum + loss
    return loss_sum

def AC_CrossEntropy(pred, yb):
    loss_sum = torch.Tensor([0.0])
    for i in range(len(yb)):
        # if torch.count_nonzero(yb[i]) >= 1:
        target_indices = (yb[i] == 1).nonzero(as_tuple=True)[0]
        result = torch.take(log_softmax(pred[i]), target_indices)
        loss = torch.mean(result).multiply(-1)
        loss_sum = loss_sum + loss
    return loss_sum


ce_loss_func = F.cross_entropy

svmguide2_ds = getTensorDataset('svmguide2')
dna_ds = getTensorDataset('dna')
vowel_ds = getTensorDataset('vowel')
segment_ds = getTensorDataset('segment')

x, y = dataset.get_imprecise_data('svmguide2', 0.3)
#print(x, y)
svmguide2_ids = getTensorImpreciseDataset('svmguide2', 0.3)


def run(k, lr, batch_size, epochs, loss_func, input_dim, hidden_dim, output_dim, ds) -> None:
    foldperf = {}
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    #name_ds = getTensorDataset(name)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(ds)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(ds, batch_size=batch_size, sampler=test_sampler)

        model, opt = get_model(lr, input_dim, hidden_dim, output_dim)

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(epochs):
            train_loss, train_correct, test_loss, test_correct = fit(model, loss_func, opt, train_loader, test_loader)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} "
                  "AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                           epochs,
                                                                           train_loss,
                                                                           test_loss,
                                                                           train_acc,
                                                                           test_acc))

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        foldperf['fold{}'.format(fold + 1)] = history

    testl_f, trainl_f, testa_f, traina_f = [], [], [], []
    for f in range(1, k + 1):
        trainl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

        traina_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t "
          "Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(np.mean(trainl_f),
                                                                            np.mean(testl_f),
                                                                            np.mean(traina_f),
                                                                            np.mean(testa_f)))




#run(10, 0.1, 8, 10, ce_loss_func, 20, 5, 3, svmguide2_ds) # overfitting occurs after 3 epochs
#run(10, 0.1, 32, 10, ce_loss_func, 180, 5, 3, dna_ds)
#run(10, 0.1, 8, 10, ce_loss_func, 10, 5, 11, vowel_ds)
#run(10, 0.1, 32, 10, ce_loss_func, 19, 5, 7, segment_ds)
run(10, 0.1, 8, 3, OSL_CrossEntropy, 20, 5, 3, svmguide2_ids) # overfitting occurs after 3 epochs
run(10, 0.1, 8, 3, AC_CrossEntropy, 20, 5, 3, svmguide2_ids)






