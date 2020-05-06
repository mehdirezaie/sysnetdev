#!/usr/bin/env python
import os
import sys
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.nn import MSELoss
from sysnet import (load_data, train_val,
                    DNN, evaluate, LRFinder, LinearNet, parse_cmd_arguments)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

''' INPUTS
'''
ap = argparse.ArgumentParser() # command line arguments
ns = parse_cmd_arguments(ap)

for (key, value) in ns.__dict__.items():
    print(f'{key}: {value}')


''' DATA
'''
dataloaders, data_len, stats = load_data(ns.input_path, ns.batch_size)


''' LEARNING RATE
'''
if ns.find_lr:
    # --- find learning rate
    fig, ax = plt.subplots()
    model = DNN(3, 20, 18, 1)
    optimizer = AdamW(params=model.parameters(),
                      lr=1.0e-7,
                      betas=(0.9, 0.999),
                      eps=1e-08,
                      weight_decay=0.01,
                      amsgrad=False)
    criterion = MSELoss() # reduction='mean'
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(dataloaders['train'], end_lr=1, num_iter=300)
    lr_finder.plot(ax=ax) # to inspect the loss-learning rate graph
    lr_finder.reset()
    fig.savefig(ns.output_path.replace('.pt', '_lr.png'), bbox_inches='tight')
    sys.exit()


''' TRAINING
'''
lr_best = 1.0e-3
eta_min = 1.0e-5
model = DNN(3, 20, 18, 1)
optimizer = AdamW(params=model.parameters(),
                  lr=lr_best,
                  betas=(0.9, 0.999),
                  eps=1e-08,
                  weight_decay=0.01,
                  amsgrad=False)
scheduler = CosineAnnealingWarmRestarts(optimizer,
                                       T_0=10,
                                       T_mult=2,
                                       eta_min=eta_min)
criterion = MSELoss() # reduction='mean'

train_losses, val_losses = train_val(model,
                                    dataloaders,
                                    data_len,
                                    criterion,
                                    optimizer,
                                    ns.nepochs,
                                    ns.output_path,
                                    device,
                                    scheduler)

plt.figure()
plt.plot(train_losses, 'b-',
         val_losses,'b--')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.savefig(ns.output_path.replace('.pt', '_loss.png'), bbox_inches='tight')
plt.close()

''' EVALUATE
'''
model = DNN(3, 20, 18, 1)
model.load_state_dict(torch.load(ns.output_path))
criterion = MSELoss() # reduction='mean'
test_loss = evaluate(model, dataloaders, data_len,
                     criterion, device, phase='test')
print(f'test loss: {test_loss:.3f}')
