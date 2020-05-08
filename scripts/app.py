#!/usr/bin/env python
import os
import sys
import argparse

import sysnet

#torch.autograd.set_detect_anomaly(True)



''' INPUTS
'''
ap = argparse.ArgumentParser() # command line arguments
ns = sysnet.parse_cmd_arguments(ap)
for (key, value) in ns.__dict__.items():
    print(f'{key}: {value}')

my_sysnet = sysnet.SYSNet(ns)
my_sysnet.run(eta_min=1.0e-5,
              lr_best=1.0e-3,
              best_structure=(3, 20, 18, 1),
              l1_alpha=1.0e-6,
              savefig=True)
