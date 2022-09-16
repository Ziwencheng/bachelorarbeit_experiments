import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import wandb
import ray
import torch
import torch.optim as optim
from ray import air, tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd


import dataset
import neural_network


def get_model(lr, input_dim, num_layers, hidden_dim, output_dim):
    model = neural_network.Neural_Network(input_dim, num_layers, hidden_dim, output_dim)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        train_loss += loss
    return train_loss


def validate(model, loss_func, val_dl):
    val_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for xb, yb in val_dl:
            loss, pred_indices = loss_batch(model, loss_func, xb, yb)
            val_loss += loss * len(xb)
            # if yb.shape
            yb_indices = torch.argmax(yb, dim=1)
            val_correct += (pred_indices == yb_indices).sum().item()
    return val_loss, val_correct

def getTensorDatasetVowel(random_state, corruption, uniform):
    lb = preprocessing.LabelBinarizer()
    lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x_train, y_train = dataset.get_precise_data("vowel")
    x, y = dataset.get_test_data("vowel")
    y = lb.transform(y)
    y = y.astype(np.float32)
    x_val, x_test, y_val, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    y_itrain = dataset.get_imprecise_data("vowel", corruption, uniform, y_train)
    xt_train = torch.from_numpy(x_train)
    yt_train = torch.from_numpy(y_itrain)
    xt_test = torch.from_numpy(x_test)
    yt_test = torch.from_numpy(y_test)
    xt_val = torch.from_numpy(x_val)
    yt_val = torch.from_numpy(y_val)
    Train_ds = TensorDataset(xt_train, yt_train)
    Test_ds = TensorDataset(xt_test, yt_test)
    Val_ds = TensorDataset(xt_val, yt_val)
    return Train_ds, Test_ds, Val_ds

def showTSNE(x, y, output_dim, name, random_state, corruption, uniform):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x)
    labelslist = convertToCategorical(y)
    df = pd.DataFrame()
    df["classes"] = labelslist
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    if uniform:
        struni = "uniform"
    else:
        struni = "skewed"
    title = name + " dataset with rs" + str(random_state) + struni + str(corruption) + "imputation T-SNE projection"
    color_dict = dict({'0': 'red',
                       '1': 'green',
                       '2': 'blue',
                       'imprecise': 'gray'})
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.classes, data=df, palette=color_dict).set(title=title)
    plt.legend(loc='upper left', fontsize='xx-small')
    my_path = os.path.abspath('/Users/ziwencheng/scatterplots_datasets')
    my_file = title + '.png'
    plt.savefig(os.path.join(my_path, my_file))

def convertToCategorical(y):
    list = []
    for i in range(len(y)):
        count = y[i].tolist().count(1)
        if count > 1:
            list.append('imprecise')
        else:
            list.append(str(y[i].tolist().index(1)))
    return list

def getTensorDataset(name, random_state, corruption, uniform, output_dim):
    x, y = dataset.get_precise_data(name)
    # random_state
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    x_subtrain, x_val, y_subtrain, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)
    y_isubtrain = dataset.get_imprecise_data(name, corruption, uniform, y_subtrain)
    showTSNE(x_subtrain, y_isubtrain, output_dim, name, random_state, corruption, uniform)
    xt_train = torch.from_numpy(x_subtrain)
    yt_train = torch.from_numpy(y_isubtrain)
    xt_test = torch.from_numpy(x_test)
    yt_test = torch.from_numpy(y_test)
    xt_val = torch.from_numpy(x_val)
    yt_val = torch.from_numpy(y_val)
    Train_ds = TensorDataset(xt_train, yt_train)
    Test_ds = TensorDataset(xt_test, yt_test)
    Val_ds = TensorDataset(xt_val, yt_val)
    return Train_ds, Test_ds, Val_ds


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


def train_without_tuning(config, epochs, input_dim, output_dim, batch_size, loss_func, name, random_state,
                         corruption, uniform):
    groupname = name + str(loss_func) + "rs" + str(random_state) + str(uniform) + "corruption" + str(corruption)
    with wandb.init(project="my-test-project", entity="ziwencheng", config=config, group=groupname):

        wandb.config = config
        # wandb.config["lossfn"] = str(loss_func)
        if name == "vowel":
            train_ds, test_ds, val_ds = getTensorDatasetVowel(random_state, corruption, uniform)
        else:
            train_ds, test_ds, val_ds = getTensorDataset(name, random_state, corruption, uniform)
        model, opt = get_model(config["lr"], input_dim, config["num_layers"], config["hidden_dim"], output_dim)

        wandb.watch(model, loss_func, log="all", log_freq=10)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            train_loss = fit(model, loss_func, opt, train_loader)
            val_loss, val_correct = validate(model, ce_loss_func, val_loader)
            wandb.log({"train_loss": train_loss / len(train_ds), "val_loss": val_loss / len(val_ds),
                       "val_accuracy": val_correct / len(val_ds) * 100}, step=epoch)


#@wandb_mixin
def train(config, epochs, input_dim, output_dim, batch_size, loss_func, train_ds, val_ds, checkpoint_dir=None):
    model, opt = get_model(config["lr"], input_dim, config["num_layers"], config["hidden_dim"], output_dim)
    """
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        opt.load_state_dict(optimizer_state)
    """

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        opt.load_state_dict(optimizer_state)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        train_loss = fit(model, loss_func, opt, train_loader)
        val_loss, val_correct = validate(model, ce_loss_func, val_loader)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (model.state_dict(), opt.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"train_loss": train_loss / len(train_ds), "val_loss": val_loss / len(val_ds),
                        "val_accuracy": val_correct / len(val_ds) * 100}, checkpoint=checkpoint)
    print("Finished Training")


"""
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), opt.state_dict()), path)
"""


# tune.report(loss=(val_loss / len(val_ds)), accuracy=(val_correct / len(val_ds) * 100))

# print("Finished Training")
# wandb.log({"train_loss": train_loss / len(train_ds), "val_loss": val_loss / len(val_ds), "val_accuracy": val_correct / len(val_ds) * 100})


def test_best_model(best_result, input_dim, output_dim, batch_size, test_ds):
    """
    wandb.init(project="my-test-project", entity="ziwencheng", group="svmguide2_re_osl_rs0_skewed",
               job_type="test_best_model")

    wandb.config["lossfn"] = "Regularized_OSL"
    wandb.config["dataset"] = "svmguide2"
    wandb.config["random_state"] = 0
    wandb.config["corruption"] = "skewed"
    """
    wandb.config["hidden_dim"] = best_result.config["hidden_dim"]
    wandb.config["lr"] = best_result.config["lr"]
    wandb.config["num_layers"] = best_result.config["num_layers"]

    best_trained_model, opt = get_model(best_result.config["lr"], input_dim, best_result.config["num_layers"],
                                        best_result.config["hidden_dim"], output_dim)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    test_loss, test_correct = validate(best_trained_model, ce_loss_func, test_loader)
    wandb.log({"test_loss": test_loss / len(test_ds), "test_accuracy": test_correct / len(test_ds) * 100})
    print("Best trial test set accuracy: {}".format(test_correct / len(test_ds) * 100))


"""
    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)
"""


# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# BayesOpt does not support parameters of type `Categorical`
# ayesopt = BayesOptSearch(metric="accuracy", mode="max", random_search_steps=4)

def main(search_space, epochs, input_dim, output_dim, batch_size, loss_func, name, random_state, uniform, corruption,
         num_samples=10, max_num_epochs=10):
    groupname = name + str(loss_func) + "rs" + str(random_state) + str(uniform) + "corruption" + str(corruption)
    run = wandb.init(project="bachelor_thesis_experiments", entity="ziwencheng", group=groupname,
                     job_type="test_best_model", reinit=True)
    """
    wandb.init(project="new_experiment", entity="ziwencheng", group=groupname,
               job_type="test_best_model")
    """
    if loss_func == OSL_CrossEntropy:
        wandb.config["lossfn"] = "OSL"
    elif loss_func == PSL_CrossEntropy:
        wandb.config["lossfn"] = "PSL"
    else:
        wandb.config["lossfn"] = "Regularized_OSL"
    wandb.config["dataset"] = name
    wandb.config["corruption"] = str(corruption)
    wandb.config["random_state"] = str(random_state)
    if uniform:
        wandb.config["uniform"] = "uniform"
    else:
        wandb.config["uniform"] = "skewed"
    """
    if name == "vowel":
        train_ds, test_ds, val_ds = getTensorDatasetVowel(random_state, corruption, uniform)
    else:
    """
    train_ds, test_ds, val_ds = getTensorDataset(name, random_state, corruption, uniform)

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=100,
        reduction_factor=2)

    hyperopt_search = HyperOptSearch(
        metric="val_accuracy", mode="max")
        # points_to_evaluate=current_best_params

    tuner = tune.Tuner(
        tune.with_parameters(train, epochs=epochs, input_dim=input_dim, output_dim=output_dim,
                             batch_size=batch_size, loss_func=loss_func, train_ds=train_ds, val_ds=val_ds),
        tune_config=tune.TuneConfig(
            # accuracy
            metric="val_accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
            search_alg=hyperopt_search
        ),
        run_config=air.RunConfig(
            callbacks=[WandbLoggerCallback(
                project="bachelor_thesis_experiments",
                group=groupname,
                api_key="4340067baf7d24002171ba53206f331b091e0f7a",
                log_config=True)]
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result("val_accuracy", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["val_loss"]))
    print("Best trial final validation accuracy: {}".format(best_result.metrics["val_accuracy"]))
    test_best_model(best_result, input_dim, output_dim, batch_size, test_ds)
    run.finish()

"""
def main(search_space, epochs, input_dim, output_dim, batch_size, name, current_best_params):
    # names = ['svmguide2', 'vowel', 'dna', 'segment']
    lossf = [OSL_CrossEntropy, PSL_CrossEntropy, Regularized_OSL]
    uniform = [True, False]
    # for name in names:
    for lf in lossf:
        for rs in range(5):
            for u in uniform:
                tuneTest(search_space, epochs, input_dim, output_dim, batch_size, lf, name, current_best_params,
                         random_state=rs, uniform=u, max_num_epochs=100, num_samples=20)

    analysis = tune.run(
        tune.with_parameters(train, epochs=epochs, input_dim=input_dim, output_dim=output_dim,
                             batch_size=batch_size, loss_func=loss_func, train_ds=train_ds, val_ds=val_ds),
        config=search_space,
        metric="val_loss",
        mode="min",
        # stop={"training_iteration": 20},
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=hyperopt_search,
        #loggers=[WandbLogger],
        callbacks=[WandbLoggerCallback(
            project="my-test-project",
            api_key="4340067baf7d24002171ba53206f331b091e0f7a",
            group="svmguide2_osl_rs0_skewed",
            log_config=True)]
    )
    best_trial = analysis.get_best_trial("loss", "min", "last")
    test_best_model(best_trial, input_dim, output_dim, batch_size, test_ds)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
"""


def run(k, lr, batch_size, epochs, loss_func, input_dim, hidden_dim, output_dim, ds) -> None:
    foldperf = {}
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    # name_ds = getTensorDataset(name)
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


# run(10, 0.1, 8, 10, ce_loss_func, 20, 5, 3, svmguide2_ds) # overfitting occurs after 3 epochs
# run(10, 0.1, 32, 10, ce_loss_func, 180, 5, 3, dna_ds)
# run(10, 0.1, 8, 10, ce_loss_func, 10, 5, 11, vowel_ds)
# run(10, 0.1, 32, 10, ce_loss_func, 19, 5, 7, segment_ds)
"""
# search_space for narrowing down the range of hyperparameters
search_space = {
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "hidden_dim": tune.randint(5, 10),
        "num_layers": tune.randint(3, 6)
}

Regularized_OSL
search_space = {
    "lr": tune.loguniform(8 * 1e-3, 2 * 1e-2),
    "hidden_dim": tune.choice([5, 6, 7]),
    "num_layers": tune.choice([2, 3]),
    # wandb configuration
    "wandb": {"api_key": "4340067baf7d24002171ba53206f331b091e0f7a", "project": "my-test-project"}
}
#this combination got the best val_accuracy for psl svmguide2 "lr": 0.015, "hidden_dim": 5, "num_layers": 3

search_space = {
    "lr": tune.loguniform(5 * 1e-3, 2 * 1e-2),
    "hidden_dim": tune.choice([5, 6, 7, 8, 9, 10]),
    "num_layers": tune.choice([3, 4, 5, 6, 7]),
    # wandb configuration
    "wandb": {"api_key": "4340067baf7d24002171ba53206f331b091e0f7a", "project": "my-test-project"}
}
search_space = {
        "lr": 0.015,
        "hidden_dim": 6,
        "num_layers": 4,
    }
current_best_params = [{'lr': 0.01, 'hidden_dim': 6, 'num_layers': 4},
                           {'lr': 0.01, 'hidden_dim': 9, 'num_layers': 6}]
# current best list for OSL
current_best_params = [{'lr': 0.0109, 'hidden_dim': 7, 'num_layers': 3},
                           {'lr': 0.0179, 'hidden_dim': 9, 'num_layers': 4}]
current_best_params = []
# search space for dataset svmguide2
search_space = {
        "lr": tune.loguniform(5 * 1e-3, 2 * 1e-2),
        "hidden_dim": tune.choice([5, 6, 7, 8, 9]),
        "num_layers": tune.choice([3, 4, 5, 6]),
        # wandb configuration
        # "wandb": {"api_key": "4340067baf7d24002171ba53206f331b091e0f7a", "project": "my-test-project"}
    }
# search space for dataset dna
search_space = {
        "lr": tune.loguniform(5 * 1e-5, 5 * 1e-4),
        "hidden_dim": tune.randint(5, 15),
        "num_layers": tune.randint(3, 10)
        # wandb configuration
        # "wandb": {"api_key": "4340067baf7d24002171ba53206f331b091e0f7a", "project": "my-test-project"}
    }
# search space for dataset segment
search_space = {
        "lr": tune.loguniform(5 * 1e-3, 2 * 1e-2),
        "hidden_dim": tune.randint(1, 15),
        "num_layers": tune.randint(1, 7)
    }
current_best_params = []
    # current_best_params = [{'lr': 0.006418290482975595, 'hidden_dim': 7, 'num_layers': 3},
    #                        {'lr': 0.008579255036167212, 'hidden_dim': 7, 'num_layers': 3}]
    main(search_space, 100, 180, 3, 32, PSL_CrossEntropy, "dna", current_best_params,
         random_state=4, uniform=False, max_num_epochs=100, num_samples=10)

# current best parameter for dataset dna  {'lr': 0.001394025543789292, 'hidden_dim': 9, 'num_layers': 3} 92%
{'lr': 0.0009179696608281584, 'hidden_dim': 14, 'num_layers': 5} 93%

# dataset vowel
search_space = {
        "lr": tune.sample_from(lambda spec: 10 ** (-100 * np.random.rand())),
        "hidden_dim": tune.randint(1, 15),
        "num_layers": tune.randint(1, 10)
    }
main(search_space, 100, 10, 11, 8, OSL_CrossEntropy, "vowel", current_best_params,
         random_state=0, uniform=True, corruption=0.3, max_num_epochs=100, num_samples=20)

train_without_tuning(search_space, 20, 10, 11, 8, OSL_CrossEntropy, "vowel", 0, 0.3, True)

main(search_space, 100, 19, 7, 32, OSL_CrossEntropy, "segment", current_best_params,
         random_state=0, uniform=True, corruption=0.3, max_num_epochs=100, num_samples=20)

"""
ce_loss_func = F.cross_entropy

if __name__ == "__main__":
    search_space = {
        "lr": tune.loguniform(1e-5, 0.5),
        "hidden_dim": tune.randint(1, 15),
        "num_layers": tune.randint(1, 7)
    }

    lossf = [OSL_CrossEntropy, PSL_CrossEntropy, Regularized_OSL]
    uniform = [True, False]
    # for name in names:
    for lf in lossf:
        for rs in range(5):
            for u in uniform:
                main(search_space, 100, 20, 3, 8, lf, "svmguide2", random_state=rs, uniform=u, corruption=0.3,
                     max_num_epochs=100, num_samples=50)

    """
    # num_samples 50
    main(search_space, 100, 10, 11, 8, Regularized_OSL, "vowel", random_state=3, uniform=True, corruption=0.3, 
         max_num_epochs=100, num_samples=50)
    """
    # num_samples 50
    """
    iris = load_iris()
    x = iris.data
    y = iris.target
    print(x)
    print(y)
   
    x, y = dataset.get_precise_data("svmguide2")
    y_im = dataset.get_imprecise_data("svmguide2", 0.3, False, y)
    list = convertToCategorical(y)
    list2 = convertToCategorical(y_im)
    print(y)
    print(list)
    print(y_im)
    print(list2)
    """
    # getTensorDataset("svmguide2", 0, 0.3, False, 3)

