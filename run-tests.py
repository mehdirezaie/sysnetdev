import unittest

from sysnet.sources.models import DNN, DNNPoisson
from torch.autograd import Variable
import torch

def test_dnn():
    """ Test DNN and DNNP """
    shape = (3, 100, 4, 1) # (# layers, # units, # input, # output)

    dnn = DNN(*shape)
    dnnp = DNNPoisson(*shape) 

    x = Variable(torch.Tensor([[1, 2, 3, 4], 
                              [5, 6, 7, 8]]))
    y_ = dnn(x)
    y1 = torch.log(1. + torch.exp(y_))
    y2 = dnnp(x)

    return torch.allclose(y1, y2)





class TestDNNs(unittest.TestCase):
    def test_allclose(self):
        self.assertTrue(test_dnn())


if __name__ == '__main__':
    unittest.main(argv=[''],verbosity=2, exit=False)
