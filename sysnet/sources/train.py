import os
import copy
import logging
import numpy as np

import torch
from torch.optim import AdamW, SGD

from .callbacks import EarlyStopping
from .io import save_checkpoint, load_checkpoint

"""
Train DL models
"""

__adamw_kwargs__ = dict(betas=(0.9, 0.999), eps=1e-08,
                        weight_decay=0.0, amsgrad=False)
__sgd_kwargs__ = dict(momentum=0.9, dampening=0, weight_decay=0)




def init_optim(optimizer):
    """

    parameters
    ----------
    optimizer : str,
        name of the optimizer

    returns
    -------
    Optim, Optim_kwargs
    """
    optimizer = optimizer.lower()
    
    if optimizer == 'adamw':
        return AdamW, __adamw_kwargs__
    
    elif optimizer == 'sgd':
        return SGD, __sgd_kwargs__
    
    else:
        raise NotImplementedError(f'{optimizer} not implemented')
    

def train(model, optimizer, loss_fn, dataloader, params, epoch, scheduler=None):
    """
    Train the model on `num_steps` batches
    parameters
    ----------
    model : (torch.nn.Module) the neural network
    optimizer : (torch.optim) optimizer for parameters of model
    loss_fn : a function that takes model output and targets and computes loss
    dataloader : (DataLoader) a torch.utils.data.DataLoader object fetches training set
    params : (Params) hyper-parameters
    epoch : (int) epoch
    scheduler : (optional) torch.optim.lr_scheduler, e.g., CosineAnnealingWarmRestarts

    returns
    -------
    loss_avg : training loss
    """
    model.train()  
    loss_avg = RunningAverage() 
    num_steps = len(dataloader)

    for i, (data, target, fpix, __) in enumerate(dataloader):
        
        data = data.to(params['device'])
        target = target.to(params['device'])
        fpix = fpix.to(params['device'])

        with torch.set_grad_enabled(True):  # only on training phase
            
            optimizer.zero_grad()  # clear previous gradients
            output = model(data)
            loss = loss_fn(output*fpix, target)
            
            if params['l1_alpha'] > 0.0: # do L1 regularization if l1 alpha is positive
                l1loss = l1_loss(model)
                loss_tot = loss + params['l1_alpha']*l1loss
            else:
                loss_tot = loss
            
            loss_tot.backward()
            optimizer.step()
            
            if scheduler is not None:  # e.g., Cosine Annealing
                scheduler.step(epoch+i/num_steps)
            
            loss_avg.update(loss, data.size(0))

    return loss_avg().item()


def evaluate(model, loss_fn, dataloader, params, return_ypred=False):
    model.eval()
    loss_avg = RunningAverage()

    if return_ypred:  # return hpix,ypred for test set
        list_hpix = []
        list_ypred = []

    with torch.no_grad():

        for (data, target, fpix, hpix) in dataloader:
            
            data = data.to(params['device'])
            target = target.to(params['device'])
            fpix = fpix.to(params['device'])

            output = model(data)
            loss = loss_fn(output*fpix, target)

            loss_avg.update(loss, data.size(0))

            if return_ypred:
                list_hpix.append(hpix)
                list_ypred.append(output)

        ret = (loss_avg().item(), )

        if return_ypred:
            hpix = torch.cat(list_hpix)
            ypred = torch.cat(list_ypred)  # .squeeze()
            ret += (hpix, ypred)

    return ret


def train_and_eval(model, optimizer, loss_fn, dataloaders, params,
                   checkpoint_path=None, scheduler=None, restore_model=None, return_losses=False):
    """
    Train and evaluate a deep learning model every epoch
    """
    epoch_first = 0
    epoch_last = params['nepochs']
    best_epoch = 0
    
    best_val_loss = 1.0e8  # a very large number
    train_losses = []     # placeholders for losses
    valid_losses = []

    if (restore_model is not None) & (checkpoint_path is not None):  # reload weights
        restore_path = os.path.join(
            checkpoint_path, restore_model + '.pth.tar')
        #logging.info(f"Restoring parameters from {restore_path}")
        checkpoint = load_checkpoint(restore_path, model, optimizer, scheduler)  
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['epoch']
        epoch_first += checkpoint['epoch']
        epoch_last += checkpoint['epoch']

    # FIXME: see https://pytorch.org/docs/stable/optim.html#constructing-it
    model = model.to(params['device'])

    if checkpoint_path is not None:
        best_model_wts = copy.deepcopy(model.state_dict())  # copy `best` model
        optim_state = copy.deepcopy(optimizer.state_dict())
        if scheduler is not None:
            scheduler_state = copy.deepcopy(scheduler.state_dict())
        else:
            scheduler_state = None

    # early_stopping = EarlyStopping(patience=10, verbose=True) # callbacks

    for epoch in range(epoch_first, epoch_last):  # training loop for `nepochs`
        msg = f"Epoch {epoch}/{epoch_last-1} "

        # one full pass over the training set
        train_loss = train(model, optimizer, loss_fn,
                           dataloaders['train'], params, epoch, scheduler=scheduler)
        train_losses.append(train_loss)
        msg += f"train loss: {train_loss:.6f} "

        # one evaluation over the validation set
        valid_loss, = evaluate(
            model, loss_fn, dataloaders['valid'], params, return_ypred=False)
        valid_losses.append(valid_loss)
        msg += f"valid loss: {valid_loss:.6f} "

        if (valid_loss < best_val_loss):
            best_val_loss = valid_loss
            
            if checkpoint_path is not None:
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())                
                optim_state = copy.deepcopy(optimizer.state_dict())
                if scheduler is not None:
                    scheduler_state = copy.deepcopy(scheduler.state_dict())
                else:
                    scheduler_state = None

        if scheduler is not None:
            msg += f"lr: {scheduler.get_last_lr()[0]:.6f}"

        if params['verbose']:
            logging.info(msg)
        # Early stopping
        # early_stopping(valid_loss)
        # if early_stopping.early_stop:
        #    print(f'!--- Early stopping at {epoch:02d}/{nepochs-1:2d} ---!')
        #    break

    if checkpoint_path is not None:
        save_checkpoint({'epoch': best_epoch,
                         'state_dict': best_model_wts,
                         'optim_dict': optim_state,
                         'scheduler_dict': scheduler_state,
                         'best_val_loss': best_val_loss},
                        checkpoint=checkpoint_path)
        #logging.info(f'saved best model at {checkpoint_path}')

    ret = (best_val_loss, )
    if return_losses:
        ret += (train_losses, valid_losses)

    return ret


def l1_loss(model):
    """ L1 regularization term """
    l1_reg = torch.norm(model.fc[0].weight, p=1)
    return l1_reg


# def add_regularization(model, loss, L1norm=True, L2norm=True, L1lambda=1.0, L2lambda=1.0):
#     if L1norm:
#         l1_reg = torch.norm(model.fc[0].weight, p=1)
#         loss += L1lambda * l1_reg

#     if L2norm:
#         l2_reg = None
#         for W in model.parameters():
#             if l2_reg is None:
#                 l2_reg = W.norm(2)
#             else:
#                 l2_reg += W.norm(2)

#         loss += L2lambda * l2_reg

#     return loss


def weight_reset(m):
    if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def tune_l1_scale(model, Optim, dataloaders, loss_fn, l1_alphas, params):
    history = {
        'l1_alphas': l1_alphas,
        'best_val_losses': []
    }
    params_ = params.copy()
    for l1_alpha in l1_alphas:

        model.apply(weight_reset)
        optimizer = Optim(params=model.parameters(), **params_['optim_kw'])

        params_.update(l1_alpha=l1_alpha)
        best_val_loss, = train_and_eval(
            model, optimizer, loss_fn, dataloaders, params_, return_losses=False)
        history['best_val_losses'].append(best_val_loss)
        logging.info(f'model with {l1_alpha} done {best_val_loss:.6f}')

    #logging.info(f'history: {history}')
    best_l1_alpha = history['l1_alphas'][np.argmin(
        history['best_val_losses'])]

    return best_l1_alpha


def tune_model_structure(Model, Optim, dataloaders, loss_fn, structures, params):
    history = {
        'structures': structures,
        'best_val_losses': []
    }
    for structure in structures:
        model = Model(*structure, seed=params['seed'])  # e.g., (3, 20, 18, 1)
        optimizer = Optim(params=model.parameters(), **params['optim_kw'])

        best_val_loss, = train_and_eval(
            model, optimizer, loss_fn, dataloaders, params, return_losses=False)
        history['best_val_losses'].append(best_val_loss)
        logging.info(f'model with {structure} done {best_val_loss:.6f}')

    #logging.info(f'history: {history}')
    best_structure = history['structures'][np.argmin(
        history['best_val_losses'])]
    return best_structure


def compute_baseline_losses(dataloaders, loss_fn):
    
    # baseline: avg. of training label
    y_train = RunningAverage()
    for _, target, _, _ in dataloaders['train']:
        y_train.update(target.sum(), target.size(0))

    pred_ = y_train()
    
    baseline_losses = {}
    
    for sample, dataloader in dataloaders.items():
        
        base_loss = RunningAverage()
        
        for _, target, fpix, _ in dataloader:
            loss_ = loss_fn(pred_*fpix, target)
            base_loss.update(loss_, target.size(0))
        
        baseline_losses[f'base_{sample}_loss'] = base_loss().item()
    
    return baseline_losses


class RunningAverage(object):
    """
    A simple class that maintains the running average of a quantity
    credit: https://github.com/cs230-stanford

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, value, step=1):
        self.total += value
        self.steps += step

    def __call__(self):
        return self.total/float(self.steps)
