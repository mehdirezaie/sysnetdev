#!/usr/bin/env python
import os
import sys
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from time import time
from torch.nn import MSELoss
from sysnet import (load_data, train_val, tune_L1, tune_model_structure,
                    DNN, evaluate, LRFinder, LinearNet, parse_cmd_arguments)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

#torch.autograd.set_detect_anomaly(True)
t0 = time()

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
dataloaders, datasets_len, stats = load_data(ns.input_path, ns.batch_size)
print(f'data loaded in {time()-t0:.3f} sec')

if ns.find_lr:
    ''' LEARNING RATE
    '''
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
    print(f'LR finder done in {time()-t0:.3f} sec')
    sys.exit()
else:
    lr_best = 1.0e-3 # manually set these two
    eta_min = 1.0e-5
print(f'lr_best: {lr_best}')
print(f'eta_min: {eta_min}')


adamw_kw = dict(lr=lr_best,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False)

if ns.find_structure:
    ''' NN structure tunning
    '''
    print('NN structure is being tunned')
    structures = [(3, 20, 18, 1), (4, 20, 18, 1), (5, 20, 18, 1)]
    criterion = MSELoss() # reduction='mean'
    best_structure = tune_model_structure(DNN,
                                          dataloaders,
                                          datasets_len,
                                          criterion,
                                          ns.nepochs,
                                          device,
                                          structures,
                                          adamw_kw=adamw_kw)

    print(f'find best structure in {time()-t0:.3f} sec')

else:
    best_structure = (5, 20, 18, 1)
print(f'best_structure: {best_structure}')

if ns.find_l1:
    ''' L1 regularization finder
    '''
    print('L1 regularization scale is being tunned')
    model = DNN(*best_structure)
    optimizer = AdamW(params=model.parameters(), **adamw_kw)
    criterion = MSELoss() # reduction='mean'
    l1_alpha = tune_L1(model,
                        dataloaders,
                        datasets_len,
                        criterion,
                        optimizer,
                        ns.nepochs,
                        device)
    print(f'find best L1 scale in {time()-t0:.3f} sec')
else:
    l1_alpha = 1.0e-6
print(f'l1_alpha: {l1_alpha}')


''' TRAINING
'''
model = DNN(*best_structure)
optimizer = AdamW(params=model.parameters(), **adamw_kw)
scheduler = CosineAnnealingWarmRestarts(optimizer,
                                       T_0=10,
                                       T_mult=2,
                                       eta_min=eta_min)
criterion = MSELoss() # reduction='mean'


train_losses, val_losses,_ = train_val(model=model,
                                    dataloaders=dataloaders,
                                    datasets_len=datasets_len,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    nepochs=ns.nepochs,
                                    device=device,
                                    output_path=ns.output_path,
                                    scheduler=scheduler)

print(f'finish training in {time()-t0:.3f} sec')

plt.figure()
plt.plot(train_losses, 'b-',
         val_losses,'b--')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.savefig(ns.output_path.replace('.pt', '_loss.png'), bbox_inches='tight')
plt.close()
print(f'make Loss vs epoch plot in {time()-t0:.3f} sec')

''' EVALUATE
'''
model = DNN(*best_structure)
model.load_state_dict(torch.load(ns.output_path))
criterion = MSELoss() # reduction='mean'
test_loss = evaluate(model=model,
                    dataloaders=dataloaders,
                    datasets_len=datasets_len,
                    criterion=criterion,
                    device=device,
                    phase='test')
print(f'finish evaluation in {time()-t0:.3f} sec')
print(f'test loss: {test_loss:.3f}')
