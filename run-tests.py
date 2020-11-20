import unittest

from sysnet.sources.models import init_model
from torch.autograd import Variable
import torch

def test_dnn():
    """ Test DNN and DNNP """
    shape = (3, 100, 4, 1) # (# layers, # units, # input, # output)
    
    DNN = init_model('dnn')
    DNNPoisson = init_model('dnnp')

    dnn = DNN(*shape)
    dnnp = DNNPoisson(*shape) 

    x = Variable(torch.Tensor([[1, 2, 3, 4], 
                              [5, 6, 7, 8]]))
    y_ = dnn(x)
    y1 = torch.log(1. + torch.exp(y_))
    y2 = dnnp(x)

    return torch.allclose(y1, y2)


def test_lin():
    """ Test lin and linp """
    shape = (4, 1) # (# input, # output)
    
    Lin = init_model('lin')
    LinP = init_model('linp')

    lin = Lin(*shape)
    linp = LinP(*shape) 

    x = Variable(torch.Tensor([[1, 2, 3, 4], 
                              [5, 6, 7, 8]]))
    y_ = lin(x)
    y1 = torch.log(1. + torch.exp(y_))
    y2 = linp(x)

    return torch.allclose(y1, y2)



class TestModels(unittest.TestCase):
    def test_DNN(self):
        self.assertTrue(test_dnn())

    def test_LIN(self):
        self.assertTrue(test_lin())

if __name__ == '__main__':
    unittest.main(argv=[''],verbosity=2, exit=False)
