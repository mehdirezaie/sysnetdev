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

    
# preprocess    

# feature selection and regression
my_sysnet = sysnet.SYSNet(ns)
my_sysnet.run(eta_min=ns.eta_min,
              lr_best=ns.lr_best,
              best_structure=ns.best_structure,
              l1_alpha=ns.l1_alpha,
              savefig=True)

# post-process