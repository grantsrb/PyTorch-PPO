import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt

"""
Description: A simple feedforward model with batchnorm.
"""

class Model(nn.Module):
    def __init__(self, input_space, output_space, hidden_dim=200, bnorm=False):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_space = input_space
        self.output_space = output_space
        self.use_bnorm = bnorm

        self.entry = nn.Linear(input_space[-1], self.hidden_dim)
        self.bnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bnorm2 = nn.BatchNorm1d(self.hidden_dim)

        self.action_out = nn.Linear(self.hidden_dim, output_space)
        self.value_out = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        fx = x.data[:,0,:] - .5*x.data[:,1,:]

        fx = F.relu(self.entry(Variable(fx)))
        if self.use_bnorm: fx = self.bnorm1(fx)
        fx = F.relu(self.hidden(fx))
        if self.use_bnorm: fx = self.bnorm2(fx)
        action = self.action_out(fx)
        value = self.value_out(fx)
        return value, action

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking into pytorch...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print("NaNs in Grad!")

    def req_grads(self, grad_on):
        """
        Used to turn off and on all gradient calculation requirements for speed.

        grad_on - bool denoting whether gradients should be calculated
        """
        for p in self.parameters():
            p.requires_grad = grad_on
