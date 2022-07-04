import torch
import torch.nn as nn
#from torch.nn import functional as F

class Neural_Network(nn.Module):
    def __init__(self, input_size, num_layers, layer_size, output_size):
        super(Neural_Network, self).__init__()
        self.layers = nn.ModuleList()
        self.tanh = nn.Tanh()
        self.layers.append(nn.Linear(input_size, layer_size))
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Linear(layer_size, layer_size))
        self.layers.append(nn.Linear(layer_size, output_size))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
            y = self.tanh(y)
        return y
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # first module
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # activation function
        self.tanh1 = nn.Tanh()
        # second module
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        # activation function
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh1(x)
        x = self.linear2(x)
        x = self.tanh2(x)
        return x
    """