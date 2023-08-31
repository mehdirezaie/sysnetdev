import torch
from torch.nn import MSELoss, PoissonNLLLoss


__all__ = ['init_loss']


def init_loss(metric):
    metric = metric.lower()
    if metric == 'mse':
        return MSELoss, {'reduction': 'none'}
    elif metric == 'pnll':
        return PoissonNLLLoss, {'log_input': False, 'reduction': 'none'}
    else:
        raise NotImplementedError(f'{metric} not implemented')


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples
