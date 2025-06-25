import matplotlib.pyplot as plt
from time import time
from .train import train_and_eval

import matplotlib
matplotlib.use('Agg')

def train_for_multiprocessing(dataloaders, nn_structure, seed,
                              Model, Optim, Scheduler, Loss, config,
                              checkpoint_path, lossfig_path, t0):
    """
    Train and evaluate a nn on training and validation sets


    parameters
    ----------
    dataloaders :
    nn_structure : (tuple of int)
    seed : (int)
    partition_id : (int)

    returns
    -------
    losses : (float, list of float, list of float)
        best validation loss
        training losses
        validation losses
    msg : (list of str)
        AJRM: Need to add a way to collect messages later.
    """
    #checkpoint_path = self.checkpoint_path_fn(partition_id, seed)
    #lossfig_path = self.lossfig_path_fn(partition_id, seed)

    model = Model(*nn_structure, seed=seed)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)
    optim_kwargs = dict(lr=config.learning_rate, **config.optim_kwargs)
    optimizer = Optim(params=model.parameters(), **optim_kwargs)
    scheduler = Scheduler(optimizer, **config.scheduler_kwargs) if Scheduler is not None else None
    loss_fn = Loss(**config.loss_kwargs)
    params = dict(nepochs=config.nepochs, device=config.device,
                  verbose=True, l1_alpha=config.l1_alpha)

    losses = train_and_eval(model, optimizer, loss_fn, dataloaders, params,
                            checkpoint_path=checkpoint_path, scheduler=scheduler,
                            restore_model=config.restore_model, return_losses=True,
                            snapshot_ensemble=config.snapshot_ensemble,
                            return_msg=False)
    plot_losses(losses, lossfig_path, config)
    #msg.append(f'finished training in {time()-t0:.3f} sec, checkout {lossfig_path}')
    #logging.info(f'finished training in {time()-t0:.3f} sec, checkout {lossfig_path}')
    return losses


def plot_losses(losses, lossfig_path, config):
    """ Plots loss vs epochs for training and validation """
    __, train_losses, val_losses = losses
    plt.figure()
    plt.plot(train_losses, 'k-', label='Training', lw=1)
    plt.plot(val_losses, 'r--', label='Validation')

    c = ['r', 'k']
    ls = ['--', '-']

    plt.legend()
    plt.ylabel(config.loss.upper())  # MSE or NPLL
    plt.xlabel('Epochs')
    plt.savefig(lossfig_path, bbox_inches='tight')
    plt.close()