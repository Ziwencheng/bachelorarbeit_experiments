
import numpy as np

from dataloader import LIBSVMLoader


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



def get_imprecise_data(name, corruption):
    loader = LIBSVMLoader(name)
    x,y = get_precise_data(name)
    S = loader.synthetic_corruption(y, corruption)
    return x, 1*S


