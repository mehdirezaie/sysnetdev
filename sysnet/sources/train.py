import copy
import torch
import os
import logging
import numpy as np
from torch.optim import AdamW
from .callbacks import EarlyStopping


__all__ = ['train_val', 'evaluate', 'tune_L1', 'tune_model_structure']

''' Train DL models

'''


def add_regularization(model, loss, L1norm=True, L2norm=True,
                        L1lambda=1.0, L2lambda=1.0):
    if L1norm:
        l1_reg = torch.norm(model.fc[0].weight, p=1)
        loss += L1lambda * l1_reg

    if L2norm:
        l2_reg = None
        for W in model.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg += W.norm(2)

        loss += L2lambda * l2_reg

    return loss


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
                'l1_lambdas':l1_lambdas,
                'best_val_losses':[]
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

    best_l1lambda = history['l1_lambdas'][np.argmin(history['best_val_losses'])]
    return best_l1lambda


def tune_model_structure(DNN,
                        dataloaders,
                        datasets_len,
                        criterion,
                        nepochs,
                        device,
                        structures,
                        adamw_kw,
                        seed=42):
    assert nepochs < 11, 'max nepochs for hyper parameter tunning: 10'
    history = {
                'structures' : structures,
                'best_val_losses':[]
                }
    for structure in structures:
        logging.info(f'model with {structure}')

        # reset model weights
        model = DNN(*structure) # e.g., (3, 20, 18, 1)

        optimizer = AdamW(params=model.parameters(), **adamw_kw)

        # call train_val ?
        _, _, best_val_loss = train_val(
                                    model=model,
                                    dataloaders=dataloaders,
                                    datasets_len=datasets_len,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    nepochs=nepochs,
                                    device=device
                                    )

        history['best_val_losses'].append(best_val_loss)

    best_structure = history['structures'][np.argmin(history['best_val_losses'])]
    return best_structure


def train_val(model,
              dataloaders,
              criterion,
              optimizer,
              nepochs,
              device,
              output_path=None,
              scheduler=None,
              L1lambda=1.0e-3,
              L2lambda=1.0e-6,
              L1norm=False,
              L2norm=False):
    '''
    Function trains the DL model, `model`


    inputs
    -------
    model: module, torch.nn.modules.module.Module
    dataloaders: dict, of torch.utils.data.dataloader
    criterion: torch.nn.modules.loss, e.g., MSELoss
    optimizer: torch.optim.optimizer.Optimizer, e.g., AdamW
    nepochs: int,
    output_path: str,
    device: torch.device, e.g., cuda or cpu
    scheduler: torch.optim.lr_scheduler, e.g., CosineAnnealingWarmRestarts


    outputs
    --------
    train_losses: list,
    valid_losses: list,
    '''
    model = model.to(device)
    if output_path is not None:
        output_dir = os.path.dirname(output_path)   # check output dir
        if not os.path.exists(output_dir):
            raise RuntimeError(f'{output_dir} does not exist')

    train_losses = [] # placeholders for losses
    valid_losses = []
    if output_path is not None:
        best_model_wts = copy.deepcopy(model.state_dict()) # `best` model
    best_val_loss = 1.0e6 # a very large number

    #--- callbacks ---
    #early_stopping = EarlyStopping(patience=10, verbose=True)

    #--- training loop `nepochs` ---
    num_iter = len(dataloaders['train']) # number of training updates
    for epoch in range(nepochs):
        running_train_loss = RunningAverage()
        running_valid_loss = RunningAverage()
        msg = f'Epoch {epoch}/{nepochs-1} '
        
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()

                i = 0
                for (data, target, _, _) in dataloaders[phase]: # training update
                    data = data.to(device)
                    target = target.to(device)
                    if scheduler is not None:
                        scheduler.step(epoch+i/num_iter)
                        i+=1
                    optimizer.zero_grad()

                    # only on training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(data)
                        loss = criterion(outputs, target)

                        if L1norm | L2norm:
                            loss = add_regularization(
                                                      model,
                                                      loss,
                                                      L1norm=L1norm,
                                                      L2norm=L2norm,
                                                      L1lambda=L1lambda,
                                                      L2lambda=L2lambda
                                                      )

                    loss.backward()
                    optimizer.step()
                    running_train_loss.update(loss.item()* data.size(0), data.size(0))

                loss_train_epoch = running_train_loss()
                train_losses.append(loss_train_epoch)
                msg += f'{phase} loss: {loss_train_epoch:.3f} '
            else:
                with torch.no_grad():
                    model.eval()
                    for (data, target, _, _) in dataloaders[phase]: # validation set
                        data = data.to(device)
                        target = target.to(device)
                        outputs = model(data)
                        loss = criterion(outputs, target)
                        running_valid_loss.update(loss.item()* data.size(0), data.size(0))

                    loss_valid_epoch = running_valid_loss()
                    valid_losses.append(loss_valid_epoch)
                    msg += f'{phase} loss: {loss_valid_epoch:.3f} '
                    if (loss_valid_epoch < best_val_loss):
                        best_val_loss = loss_valid_epoch
                        if output_path is not None:
                            best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            msg += f'lr: {scheduler.get_last_lr()[0]:.6f}' # or {scheduler.get_lr()[0]:.6f}

        logging.info(msg)
        # Early stopping
        # early_stopping(loss_valid_epoch)
        # if early_stopping.early_stop:
        #    print(f'!--- Early stopping at {epoch:02d}/{nepochs-1:2d} ---!')
        #    break

    if output_path is not None:
        if os.path.exists(output_path):
            raise RuntimeError(f'{output_path} already exists!')
        torch.save(best_model_wts, output_path)
        logging.info(f'save model at {output_path}')
    return train_losses, valid_losses, best_val_loss


def evaluate(model,
            dataloaders,
            criterion,
            device,
            phase='test'):
    # should I 
    model = model.to(device)
    
    #list_target = []
    list_hpix = []
    list_prediction = []
    
    with torch.no_grad():
        model.eval()
        loss = RunningAverage()
        for (data, target, hpix, _) in dataloaders[phase]:
            data = data.to(device)
            target = target.to(device)
            prediction = model(data)
            loss.update(criterion(prediction, target).item() * data.size(0), data.size(0))
            
            #list_target.append(target)
            list_hpix.append(hpix)
            list_prediction.append(prediction)
            
    hpix = torch.cat(list_hpix)
    pred = torch.cat(list_prediction).squeeze()
    test_loss = loss()
        
    return test_loss, hpix, pred



class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
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
    
    def update(self, val, step):
        self.total += val
        self.steps += step
    
    def __call__(self):
        return self.total/float(self.steps)
