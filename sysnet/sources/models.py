''' Implementations for Regression

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


import logging

__all__ = ['DNN', 'LinearRegression', 'LinearNet']


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
    def __init__(self, nb_layers, nb_units, input_dim, output_dim):
        assert nb_layers >= 2
        super(DNN, self).__init__()
        self.fc = nn.ModuleList()#[]
        self.bn = nn.ModuleList()#[]

        self.nb_layers = nb_layers
        for i in range(nb_layers):
            if i == 0: # input layer
                self.fc.append(nn.Linear(input_dim, nb_units))
                self.bn.append(nn.BatchNorm1d(nb_units))

            elif i == nb_layers-1: # output layer
                self.fc.append(nn.Linear(nb_units, output_dim))

            else: # hidden layers
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

class LinearNet(nn.Module):

    def __init__(self, in_features=18):
        super(LinearNet, self).__init__()
        self.hl1 = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.hl1(x)
        return x



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

    def __init__(self, reduction='sum'):
        self.logger.info(f'reduction: {reduction}')
        self.cost = torch.nn.MSELoss(reduction=reduction)

    def fit(self, x, y):
        # check input
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        # training
        xx = torch.matmul(x.T, x)
        xx_inv = torch.inverse(xx)
        self.coef_ = torch.matmul(xx_inv, torch.matmul(x.T, y))

        # cost
        cost_ = self.cost(y, self.predict(x))
        self.logger.info('training cost: {cost_:.3f}')
        self.logger.info(f'coefs: {self.coef_} cost: {cost_:.3f}')

    def predict(self, x):
        return torch.matmul(self.coef_, x.T)