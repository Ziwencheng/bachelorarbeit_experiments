import torch
import torch.nn as nn

class Neural_Network(nn.Module):
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