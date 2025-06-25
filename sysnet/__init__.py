from .sources.io import Config
from .cli import parse_cmd_arguments
from .lab import SYSNet, SYSNetSnapshot, TrainedModel, SYSNetMultiProcess, SYSNetMPI


def detect_anomaly():
    import torch
    torch.autograd.set_detect_anomaly(True) # check 