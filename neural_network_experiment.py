import numpy

import dataset
from dataloader import LIBSVMLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
import math
import wandb
import neural_network

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
import os



def get_model(lr,input_dim, num_layers, hidden_dim, output_dim):
    model = neural_network.Neural_Network(input_dim, num_layers, hidden_dim, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    return model, optimizer

def loss_batch(model, loss_func, xb, yb, opt=None):
    pred = model(xb.float())
    loss = loss_func(pred, yb)
    pred_indices = torch.argmax(pred, dim=1)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), pred_indices

def fit(model, loss_func, opt, train_dl):
        train_loss = 0.0
        model.train()
        for xb, yb in train_dl:
            loss, pred_indices = loss_batch(model, loss_func, xb, yb, opt)
            train_loss += loss * len(xb)
        return train_loss

def validate(model, loss_func, val_dl):
        val_loss, val_correct = 0.0, 0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_dl:
                loss, pred_indices = loss_batch(model, loss_func, xb, yb)
                val_loss += loss * len(xb)
                yb_indices = torch.argmax(yb, dim=1)
                val_correct += (pred_indices == yb_indices).sum().item()
        return val_loss, val_correct

"""
def getTensorDataset(name):
    x, y = dataset.get_precise_data(name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    xt_train = torch.from_numpy(x_train)
    xt_test = torch.from_numpy(x_test)
    yt_train = torch.from_numpy(y_train)
    yt_test = torch.from_numpy(y_test)
    #name_ds = TensorDataset(xt, yt)
    nameTrain_ds = TensorDataset(xt_train, yt_train)
    nameTest_ds = TensorDataset(xt_test, yt_test)
    return nameTrain_ds, nameTest_ds
"""

def getTensorDataset(x_train, x_test, y_train, y_test):
    xt_train = torch.from_numpy(x_train)
    xt_test = torch.from_numpy(x_test)
    yt_train = torch.from_numpy(y_train)
    yt_test = torch.from_numpy(y_test)
    Train_ds = TensorDataset(xt_train, yt_train)
    Test_ds = TensorDataset(xt_test, yt_test)
    return Train_ds, Test_ds

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def OSL_CrossEntropy(pred, yb):
    loss_sum = torch.Tensor([0.0])
    for i in range(len(yb)):
        target_indices = (yb[i] == 1).nonzero(as_tuple=True)[0]
        result = torch.take(log_softmax(pred[i]), target_indices)
        loss = torch.max(result).multiply(-1)
        loss_sum = loss_sum + loss
    return loss_sum

def PSL_CrossEntropy(pred, yb):
    loss_sum = torch.Tensor([0.0])
    for i in range(len(yb)):
        target_indices = (yb[i] == 1).nonzero(as_tuple=True)[0]
        result = torch.take(log_softmax(pred[i]), target_indices)
        loss = torch.min(result).multiply(-1)
        loss_sum = loss_sum + loss
    return loss_sum

def Regularized_OSL(pred, yb):
    loss_sum = torch.Tensor([0.0])
    for i in range(len(yb)):
        target_indices = (yb[i] == 1).nonzero(as_tuple=True)[0]
        result = torch.take(log_softmax(pred[i]), target_indices)
        loss = (torch.max(result) - torch.max(log_softmax(pred[i]))).multiply(-1)
        loss_sum = loss_sum + loss
    return loss_sum

ce_loss_func = F.cross_entropy

"""
dna_ds = getTensorDataset('dna')
vowel_ds = getTensorDataset('vowel')
segment_ds = getTensorDataset('segment')
"""
#x, y = dataset.get_imprecise_data('svmguide2', 0.3)
#svmguide2_ids = getTensorImpreciseDataset('svmguide2', 0.3)

search_space = {
    "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    "hidden_dim": tune.randint(5, 10),
    "num_layers": tune.randint(2, 5)
}

algo = BayesOptSearch(random_search_steps=4)

def train(config, k, epochs, input_dim, output_dim, batch_size, loss_func, train_ds, val_ds, checkpoint_dir=None):

    model, opt = get_model(config["lr"], input_dim, config["num_layers"], config["hidden_dim"], output_dim)

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        opt.load_state_dict(optimizer_state)

    train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,shuffle=True)

    for epoch in range(epochs):
        train_loss = fit(model, loss_func, opt, train_loader)
        val_loss, val_correct = validate(model, loss_func, val_loader)

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), opt.state_dict()), path)

        tune.report(loss=(val_loss / len(val_ds)), accuracy=(val_correct / len(val_ds) * 100))
    print("Finished Training")

def test_best_model(best_trial, input_dim, output_dim, name, batch_size, loss_func, train_ds, test_ds):
    model, opt = get_model(best_trial.config["lr"], input_dim, best_trial.config["num_layers"],
                           best_trial.config["hidden_dim"], output_dim)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)

    #train_ds, test_ds = getTensorDataset(name)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    for i in range(10):
        train_loss = fit(model, loss_func, opt, train_loader)
        test_loss, test_correct = validate(model, loss_func, test_loader)
        """
        metrics = {"train/train_loss": train_loss / len(train_ds),
                   "train/train_acc": train_correct / len(train_ds) * 100,
                   "train/epoch": i}
        test_metrics = {"test/test_loss": test_loss / len(test_ds),
                       "test/test_acc": test_correct / len(test_ds) * 100,
                       "test/epoch": i}
        wandb.log({**metrics, **test_metrics})
        """
    print("Best trial test set accuracy: {}".format(test_correct/len(test_ds) * 100))

def main(k, epochs, input_dim, output_dim, batch_size, loss_func, name, num_samples=10, max_num_epochs=10):
    #wandb.init(project="my-test-project", entity="ziwencheng")
    x, y = dataset.get_precise_data(name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_subtrain, x_val, y_subtrain, y_val = train_test_split(x_train, y_train, test_size=0.2)
    y_isubtrain = dataset.get_imprecise_data(name, 0.3, True, y_subtrain)
    # the subtraining dataset is imprecise, and the validation set is precise
    subTrain_ds, val_ds = getTensorDataset(x_subtrain, x_val, y_isubtrain, y_val)
    y_ival = dataset.get_imprecise_data(name, 0.3, True, y_val)
    y_itrain = np.concatenate((y_isubtrain, y_ival), axis=0)
    # the training dataset is imprecise
    train_ds, test_ds = getTensorDataset(x_train, x_test, y_itrain, y_test)

    search_space = {
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "hidden_dim": tune.randint(5, 10),
        "num_layers": tune.randint(2, 5)
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    analysis = tune.run(
        tune.with_parameters(train, k=k, epochs=epochs, input_dim=input_dim, output_dim=output_dim,
                             batch_size=batch_size, loss_func=loss_func, train_ds=subTrain_ds, val_ds=val_ds),
        config=search_space,
        metric="loss",
        mode="min",
        # search_alg=algo,
        # stop={"training_iteration": 20},
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    #print("Best trial final validation accuracy: {}".format(
        #best_trial.last_result["accuracy"]))

    if ray.util.client.ray.is_connected():
        from ray.util.ml_utils.node import force_on_current_node
        remote_fn = force_on_current_node(ray.remote(test_best_model))
        ray.get(remote_fn.remote(best_trial))
    else:
        test_best_model(best_trial, input_dim, output_dim, name, batch_size, loss_func, train_ds, test_ds)

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
#run(10, 0.1, 8, 3, OSL_CrossEntropy, 20, 5, 3, svmguide2_ids) # overfitting occurs after 3 epochs
#run(10, 0.1, 8, 3, AC_CrossEntropy, 20, 5, 3, svmguide2_ids)
#analysis = tune.run(train(10, search_space, 10, 20, 3, 8, ce_loss_func, 'svmguide2'), config=search_space)
"""
analysis2 = tune.run(
    tune.with_parameters(train, k=10, epochs=10, input_dim=180, output_dim=3, batch_size=32, loss_func=ce_loss_func,
                         name='dna'),
    config=search_space,
    mode="max",
    metric="mean_accuracy",
    #search_alg=algo,
    #stop={"training_iteration": 20},
    num_samples=10
)
best_config = analysis.best_config
#print("best config: ", analysis.get_best_config(metric="mean_accuracy", mode="max"))
#print("best config: ", analysis2.get_best_config(metric="mean_accuracy", mode="max"))
#print(best_config)
"""
if __name__ == "__main__":
    main(10, 10, 20, 3, 8, OSL_CrossEntropy, "svmguide2")
    main(10, 10, 20, 3, 8, Regularized_OSL, "svmguide2")
#k, epochs, input_dim, output_dim, batch_size, loss_func, name, num_samples=10, max_num_epochs=10
