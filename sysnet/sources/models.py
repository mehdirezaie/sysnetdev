''' Implementations for Regression

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import logging

__all__ = ['DNN', 'LinearRegression', 'LinearNet', 'DNNPoisson', 'init_model']


def init_model(model):
    if model == 'dnn':
        return DNN
    elif model == 'dnnp':
        return DNNPoisson
    elif model == 'lin':
        return LinearNet
    elif model == 'linp':
        return LinearPoisson
    else:
        raise NotImplementedError(f'Model {model} is not implemented!')


class DNN(nn.Module):
    '''
    credit: https://discuss.pytorch.org/u/matthew_zeng/

    TODO: take a list of hidden layer neurons

    examples
    ---------
    a = DNN(3, 100, 500, 500)
    input = Variable(torch.Tensor(10, 500))
    output = a(input)
    '''

    def __init__(self, nb_layers, nb_units, input_dim=18, output_dim=1, seed=42):
        assert nb_layers > 1
        torch.manual_seed(seed=seed)
        super(DNN, self).__init__()
        self.fc = nn.ModuleList()  # []
        self.bn = nn.ModuleList()  # []

        self.nb_layers = nb_layers
        for i in range(nb_layers):
            if i == 0:  # input layer
                self.fc.append(nn.Linear(input_dim, nb_units))
                self.bn.append(nn.BatchNorm1d(nb_units))

            elif i == nb_layers-1:  # output layer
                self.fc.append(nn.Linear(nb_units, output_dim))

            else:  # hidden layers
                self.fc.append(nn.Linear(nb_units, nb_units))
                self.bn.append(nn.BatchNorm1d(nb_units))

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.fc[i](x)
            else:
                x = F.relu(self.fc[i](x))
                x = self.bn[i](x)
        return x


class DNNPoisson(DNN):
    '''
    credit: https://discuss.pytorch.org/u/matthew_zeng/

    TODO: take a list of hidden layer neurons

    examples
    ---------
    a = DNNPoisson(3, 100, 500, 500)
    input = Variable(torch.Tensor(10, 500))
    output = a(input)
    '''

    def __init__(self, nb_layers, nb_units, input_dim=18, output_dim=1, seed=42):
        super(DNNPoisson, self).__init__(nb_layers, nb_units,
                                         input_dim=input_dim, output_dim=output_dim,
                                         seed=seed)

    def forward(self, x):
        x = super(DNNPoisson, self).forward(x)
        return F.softplus(x, threshold=1000)
        #for i in range(self.nb_layers):
        #    if i == self.nb_layers-1:
        #        x = self.fc[i](x)
        #        x = F.softplus(x, threshold=1000)
        #    else:
        #        x = F.relu(self.fc[i](x))
        #        x = self.bn[i](x)
        #return x


class LinearNet(nn.Module):

    def __init__(self, input_dim=18, seed=42):
        torch.manual_seed(seed=seed)
        super(LinearNet, self).__init__()
        self.hl1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.hl1(x)
        return x


class LinearPoisson(LinearNet):
    def __init__(self, input_dim=18, seed=42):
        super(LinearPoisson, self).__init__(input_dim=input_dim, seed=seed)

    def forward(self, x):
        x = super(LinearPoisson, self).forward(x)
        return F.softplus(x, threshold=1000)



class LinearRegression:
    '''
        Linear Regression with PyTorch


        Algorithm
        -----------
        thetas = (X^t.X)^-1.(X.Y)

        see e.g.,
        https://pytorch.org/docs/master/torch.html#torch.inverse
        https://pytorch.org/docs/master/nn.html#torch.nn.MSELoss

    '''

    logger = logging.getLogger('LinearRegression')

    def __init__(self, reduction='mean', add_bias=False):
        #self.logger.info(f'reduction: {reduction}')
        self.cost = torch.nn.MSELoss(reduction=reduction)
        self.add_bias = add_bias

    def fit(self, x, y):
        # check input
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        if self.add_bias:
            x = torch.cat((torch.ones(x.shape[0]).unsqueeze(1), x), 1)

        # training
        xx = torch.matmul(x.T, x)
        xx_inv = torch.inverse(xx)
        self.coef_ = torch.matmul(xx_inv, torch.matmul(x.T, y))

        # cost
        cost_ = self.evaluate(x, y, has_bias=True)
        #self.logger.info(f'training cost: {cost_:.3f}')
        #self.logger.info(f'coefs: {self.coef_} cost: {cost_:.3f}')

    def evaluate(self, x, y, has_bias=False):
        return self.cost(y, self.predict(x, has_bias)).item()

    def predict(self, x, has_bias=False):
        if (not has_bias) & self.add_bias:
            x = torch.cat((torch.ones(x.shape[0]).unsqueeze(1), x), 1)
        return torch.matmul(x, self.coef_)
