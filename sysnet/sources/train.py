import os
import copy
import logging
import numpy as np

import torch
from torch.optim import AdamW

from .callbacks import EarlyStopping
from .io import save_checkpoint, load_checkpoint

"""
Train DL models
"""


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
    # model.to(params['device'])
    model.train()  # set model to training mode
    loss_avg = RunningAverage()  # running average object for loss
    num_steps = len(dataloader)

    for i, (data, target, fpix, __) in enumerate(dataloader):

        # move to GPU
        data = data.to(params['device'])
        target = target.to(params['device'])
        fpix = fpix.to(params['device'])

        if scheduler is not None:  # e.g., Cosine Annealing
            scheduler.step(epoch+i/num_steps)

        with torch.set_grad_enabled(True):  # only on training phase

            outputs = model(data)*fpix.unsqueeze(-1)
            loss = loss_fn(outputs, target)
            
            if params['l1_alpha'] > 0.0: # do L1 regularization if l1 alpha is positive
                l1loss = l1_loss(model)
                loss += params['l1_alpha']*l1loss

            optimizer.zero_grad()  # clear previous gradients
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item(), data.size(0))

    return loss_avg()


def evaluate(model, loss_fn, dataloader, params, return_ypred=False):
    # model.to(params['device'])
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

            outputs = model(data)
            loss = loss_fn(outputs*fpix.unsqueeze(-1), target)

            loss_avg.update(loss.item(), data.size(0))

            if return_ypred:
                list_hpix.append(hpix)
                list_ypred.append(outputs)

        ret = (loss_avg(), )

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
    epoch_first,  epoch_last = 0, params['nepochs']

    if (restore_model is not None) & (checkpoint_path is not None):  # reload weights
        restore_path = os.path.join(
            checkpoint_path, restore_model + '.pth.tar')
        #logging.info(f"Restoring parameters from {restore_path}")
        checkpoint = load_checkpoint(restore_path, model, optimizer, scheduler)

        epoch_first += checkpoint['epoch']
        epoch_last += checkpoint['epoch']

    model = model.to(params['device'])

    if checkpoint_path is not None:
        best_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())  # copy `best` model
        optim_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(
            scheduler.state_dict()) if scheduler is not None else None

    best_val_loss = 1.0e8  # a very large number
    train_losses = []     # placeholders for losses
    valid_losses = []

    # early_stopping = EarlyStopping(patience=10, verbose=True) # callbacks

    for epoch in range(epoch_first, epoch_last):  # training loop for `nepochs`
        msg = f"Epoch {epoch}/{epoch_last-1} "

        # one full pass over the training set
        train_loss = train(model, optimizer, loss_fn,
                           dataloaders['train'], params, epoch, scheduler=scheduler)
        train_losses.append(train_loss)
        msg += f"train loss: {train_loss:.3f} "

        # one evaluation over the validation set
        valid_loss, = evaluate(
            model, loss_fn, dataloaders['valid'], params, return_ypred=False)
        valid_losses.append(valid_loss)
        msg += f"valid loss: {valid_loss:.3f} "

        if (valid_loss < best_val_loss):
            best_val_loss = valid_loss
            if checkpoint_path is not None:
                best_epoch = epoch + 1
                optim_state = copy.deepcopy(optimizer.state_dict())
                best_model_wts = copy.deepcopy(model.state_dict())
                scheduler_state = copy.deepcopy(
                    scheduler.state_dict()) if scheduler is not None else None

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
                         'scheduler_dict': scheduler_state},
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


def tune_L1(model,
            dataloaders,
            datasets_len,
            criterion,
            optimizer,
            nepochs,
            device):

    assert nepochs < 11, 'max nepochs for hyper parameter tunning: 10'
    l1_lambdas = np.logspace(-6, 0, 10)
    history = {
        'l1_lambdas': l1_lambdas,
        'best_val_losses': []
    }
    for l1_lambda in l1_lambdas:
        logging.info(f'l1_lambda: {l1_lambda}')
        # reset model weights
        model.apply(weight_reset)
        # call train_val ?
        _, _, best_val_loss = train_val(
            model=model,
            dataloaders=dataloaders,
            datasets_len=datasets_len,
            criterion=criterion,
            optimizer=optimizer,
            nepochs=nepochs,
            device=device,
            L1lambda=l1_lambda,
            L1norm=True
        )

        history['best_val_losses'].append(best_val_loss)

    best_l1lambda = history['l1_lambdas'][np.argmin(
        history['best_val_losses'])]
    return best_l1lambda


def tune_l1_scale(model, dataloaders, loss_fn, l1_alphas, params):
    history = {
        'l1_alphas': l1_alphas,
        'best_val_losses': []
    }
    params_ = params.copy()
    for l1_alpha in l1_alphas:

        model.apply(weight_reset)
        optimizer = AdamW(params=model.parameters(), **params_['adamw_kw'])

        params_.update(l1_alpha=l1_alpha)
        best_val_loss, = train_and_eval(
            model, optimizer, loss_fn, dataloaders, params_, return_losses=False)
        history['best_val_losses'].append(best_val_loss)
        logging.info(f'model with {l1_alpha} done {best_val_loss:.6f}')

    #logging.info(f'history: {history}')
    best_l1_alpha = history['l1_alphas'][np.argmin(
        history['best_val_losses'])]

    return best_l1_alpha


def tune_model_structure(Model, dataloaders, loss_fn, structures, params):
    history = {
        'structures': structures,
        'best_val_losses': []
    }
    for structure in structures:
        model = Model(*structure, seed=params['seed'])  # e.g., (3, 20, 18, 1)
        optimizer = AdamW(params=model.parameters(), **params['adamw_kw'])

        best_val_loss, = train_and_eval(
            model, optimizer, loss_fn, dataloaders, params, return_losses=False)
        history['best_val_losses'].append(best_val_loss)
        logging.info(f'model with {structure} done {best_val_loss:.6f}')

    #logging.info(f'history: {history}')
    best_structure = history['structures'][np.argmin(
        history['best_val_losses'])]
    return best_structure


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
