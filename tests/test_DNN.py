from sysnet.sources.models import DNN, DNNPoisson
from torch import Tensor
from torch.autograd import Variable

import torch

msg = 'Test Deep Neural Network\n'
msg += f'PyTorch: {torch.__version__}\n'
print(msg)

a = DNNPoisson(3, 100, 500, 500)

input = Variable(Tensor(10, 500))
output = a(input)

msg = f'a: {a}\n'
msg += f'input: {input}\n'
msg += f'output: {output}\n'
msg += f'a: {a.fc[0].weight}'

print(msg)
