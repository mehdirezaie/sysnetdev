
from .lab import SYSNet
from .cli import parse_cmd_arguments
from .sources.io import Config


def detect_anomaly():
    import torch
    torch.autograd.set_detect_anomaly(True) # check 