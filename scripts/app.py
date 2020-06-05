#!/usr/bin/env python
import os
import sys
import argparse

import sysnet

#torch.autograd.set_detect_anomaly(True)

ap = argparse.ArgumentParser()
ns = sysnet.parse_cmd_arguments(ap)    


# preprocess    

# feature selection and regression
pipeline = sysnet.SYSNet(ns)
pipeline.load()
pipeline.run(eta_min=ns.eta_min,
             lr_best=ns.lr_best,
             best_structure=ns.best_structure,
             l1_alpha=ns.l1_alpha,
             savefig=True)

# post-process