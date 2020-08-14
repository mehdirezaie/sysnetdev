
from .lab import SYSNet
from .cli import parse_cmd_arguments
from .sources.io import Config


def test_torch():
    import torch
    torch.autograd.set_detect_anomaly(True) # check 