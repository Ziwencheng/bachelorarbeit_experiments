
import numpy as np

from dataloader import LIBSVMLoader
from sklearn.model_selection import train_test_split


#names = LIBSVMLoader.datasets
#names = ['svmguide2', 'vowel', 'dna', 'segment']
# names = ['svmguide2', 'svmguide4', 'glass', 'iris', 'vowel', 'wine', 'vehicle']

corruptions = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]  # corruptions to test

"""
for name in names:
    print(name)

    loader = LIBSVMLoader(name)
    x, y = loader.get_trainset()


    index_to_corrupt = np.argmax(y.sum(axis=0))

    #S = np.empty((*y.shape, len(corruptions)))

    #S = np.empty(*y.shape, dtype=int)
    #print(*y.shape)
    S = loader.synthetic_corruption(y, 0.1)
    print(1*S)
    #for i, corruption in enumerate(corruptions):
#         S[..., i] = loader.skewed_corruption(y, corruption, index_to_corrupt)
        #S[..., i] = loader.synthetic_corruption(y, corruption)
        #print(S[..., i])
        
x, y = get_precise_data("dna")
print(x, y)
print(x.shape)
print(y.shape)
"""

def get_precise_data(name):
    loader = LIBSVMLoader(name)
    return loader.get_trainset()



def get_imprecise_data(name, corruption, balanced, imprecise_val):
    loader = LIBSVMLoader(name)
    x,y = get_precise_data(name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    if imprecise_val:
        if balanced:
            S = loader.synthetic_corruption(y_train, corruption)
        else:
            index_to_corrupt = np.argmax(y.sum(axis=0))
            S = loader.skewed_corruption(y_train, corruption, index_to_corrupt)
        return x_train, x_test, 1 * S, y_test

    else:
        x_subtrain, x_val, y_subtrain, y_val = train_test_split(x_train, y_train, test_size=0.2)
        if balanced:
            S = loader.synthetic_corruption(y_subtrain, corruption)
        else:
            index_to_corrupt = np.argmax(y.sum(axis=0))
            S = loader.skewed_corruption(y_subtrain, corruption, index_to_corrupt)
            return x_subtrain, x_val, 1 * S, y_val

def get_imprecise_data(name, corruption, balanced, data):
    loader = LIBSVMLoader(name)
    x, y = get_precise_data(name)
    if balanced:
        S = loader.synthetic_corruption(data, corruption)
    else:
        index_to_corrupt = np.argmax(y.sum(axis=0))
        S = loader.skewed_corruption(data, corruption, index_to_corrupt)
    return 1 * S


