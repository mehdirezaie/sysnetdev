from sysnet.models import DNN
from torch import Tensor
from torch.autograd import Variable

a = DNN(3, 100, 500, 500)

input = Variable(Tensor(10, 500))
output = a(input)

msg = f'a: {a}\n'
msg += f'input: {input}\n'
msg += f'output: {output}\n'
msg += f'a: {a.fc[0].weight}'

print(msg)
