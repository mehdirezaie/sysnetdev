import os
import logging
from time import time
import numpy as np

import torch
import sysnet.sources as src

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# set some global variables which do not change
__global_seed__ = 85        
__adamw_kwargs__ = dict(betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
__cosann_warmup_kwargs__ = dict(T_0=10, T_mult=2)
__nepochs_hyperparams__ = 2
__seed_max__ = 4294967295 # i.e., 2**32 - 1, maximum number in numpy

class SYSNet:
    """
    Implementation of a multilayer neural network 
    for mitigation of observational systematics
    """
    logger = logging.getLogger()

    def __init__(self, config):
        """
        Initializes SYSNet

        parameters
        ----------
        config : argparse.ArgumentParser object
            input_path: (str) path to input tabulated data
            output_path: (str) path to outputs
            restore_model: (str) name of the model to store weights
            batch_size: (int) size of the batch
            nepochs: (int) number of training epochs
            find_lr: (bool) find learning rate
            find_structure: False
            find_l1: False
            do_kfold: True
            normalization: z-score
            loss: mse
            model: dnn
            axes: [0, 1, 2, 3, 4, 5]
            do_rfe: False
            eta_min: 1e-05
            lr_best: 0.001
            best_structure: (4, 20)
            l1_alpha: 0.001
            zbins: [0.8, 2.2, 3.5]
            nside: 512            
        """
        self.t0 = time()
        self.config = config

        log_path = os.path.join(self.config.output_path, 'train.log')
        src.set_logger(log_path)    
        self.logger.info(f"logging in {log_path}")
                    
        self.Loss, self.config.loss_kwargs = src.init_loss(self.config.loss)
        self.collector = src.SYSNetCollector()
        self.Model = src.init_model(self.config.model)
        self.config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        self.logger.info('# --- inputs params ---')
        for (key, value) in self.config.__dict__.items():
            self.logger.info(f'{key}: {value}')
        self.logger.info(f"pipeline initialized in {time()-self.t0:.3f} s")
           
        self.ld = src.MyDataLoader(self.config.input_path, 
                                   do_kfold=self.config.do_kfold,
                                   seed=__global_seed__)
        self.logger.info(f'data loaded in {time()-self.t0:.3f} sec')
             
    def run(self):

        self.logger.info('# running pipeline ...')

        if self.config.do_rfe:
            self.axes_from_rfe = self.__run_rfe(self.config.axes)
        
        self.logger.info('# training and evaluation')
        num_partitions = 5 if self.config.do_kfold else 1

        for partition_id in range(num_partitions): # k-fold validation loop

            axes = self.__axes_for_partition(partition_id)
            nn_structure = (*self.config.nn_structure, len(axes), 1) # (# units, # hidden layers, # input layer units, # output unit)
            self.logger.info(f'partition_{partition_id} with {nn_structure}')

            dataloaders, stats = self.ld.load_data(batch_size=self.config.batch_size,
                                                   partition_id=partition_id,
                                                   normalization=self.config.normalization,
                                                   axes=axes,
                                                   loss_fn=self.Loss(**self.config.loss_kwargs)) # takes loss_fn for baseline metrics
            
            
            nn_structure = self.__tune_hyperparams(dataloaders, nn_structure, partition_id) # only once

            self.__train_and_eval_chains(dataloaders, nn_structure, partition_id, stats) # for 'nchains' times

        save = False
        if save:
            pass
            ## rewrite as a train_chain function that takes `num_chains` and returns 'num_chains' ypreds
            ## and finally exports the ypreds to .fits
            # metrics, hpind, pred = self.__run(eta_min=eta_min,
            #                                 lr_best=lr_best,
            #                                 best_structure=structure,
            #                                 l1_alpha=l1_alpha,
            #                                 savefig=savefig,
            #                                 seed=seed)
            # self.collector.collect(key, {**metrics, **self.stats}, hpind, pred)            
            # self.config.output_path = output_path_org            
            # self.collector.save(self.config.output_path.replace('.pt', '.json'), self.config.configide)

    def __train_and_eval_chains(self, dataloaders, nn_structure, partition_id, stats):
        """ Train and evaluate for 'nchain' times """
        np.random.seed(__global_seed__)
        seeds = np.random.randint(0, __seed_max__, size=self.config.nchains) # initialization seed        

        for chain_id in range(self.config.nchains): 

            seed = seeds[chain_id]
            self.logger.info(f'# running training and evaluation with seed: {seed}')

            train_val_losses = self.__train(dataloaders, nn_structure, seed, partition_id, stats)
        
            restore_model = 'best' # restore the best model
            restore_path = os.path.join(self.config.output_path, f'model_{partition_id}_{seed}', f'{restore_model}.pth.tar')
            predictions = self.__evaluate(dataloaders['test'], nn_structure, restore_path)
            self.logger.info(f'best val loss: {train_val_losses[0]:.3f}, test loss: {predictions[0]:.3f}')

            print(predictions[1])
            print(predictions[2])
            print(train_val_losses[1])
            print(train_val_losses[2])
            print('\n')

    def __train(self, dataloaders, nn_structure, seed, partition_id, stats):
        """ train and evaluate a nn on training and validation sets """
        checkpoint_path = os.path.join(self.config.output_path, f'model_{partition_id}_{seed}') 
        lossfig_path = os.path.join(checkpoint_path, f'loss_model_{partition_id}_{seed}.png')

        model = self.Model(*nn_structure, seed=seed)
        adamw_kwargs = dict(lr=self.config.learning_rate, **__adamw_kwargs__)
        optimizer = src.AdamW(params=model.parameters(), **adamw_kwargs)
        scheduler = src.CosineAnnealingWarmRestarts(optimizer, eta_min=self.config.eta_min, **__cosann_warmup_kwargs__)
        loss_fn = self.Loss(**self.config.loss_kwargs)     
        params = dict(nepochs=self.config.nepochs, device=self.config.device, verbose=True)  

        losses = src.train_and_eval(model, optimizer, loss_fn, dataloaders, params, 
                                    checkpoint_path=checkpoint_path, scheduler=scheduler, 
                                    restore_model=self.config.restore_model, return_losses=True)
        self.__plot_losses(losses, stats, lossfig_path)
        self.logger.info(f'finished training in {time()-self.t0:.3f} sec, checkout {lossfig_path}')
        return losses

    def __evaluate(self, dataloader, nn_structure, restore_path):
        """ Evaluates a trained neural network on the test set

        see also
        --------
        1. https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
        """
        self.logger.info(f"Restoring parameters from {restore_path}")

        params = dict(device=self.config.device)
        model = self.Model(*nn_structure)
        src.load_checkpoint(restore_path, model)
        model = model.to(params['device'])
        loss_fn = self.Loss(**self.config.loss_kwargs)

        predictions = src.evaluate(model, loss_fn, dataloader, params, return_ypred=True)
        self.logger.info(f'finish evaluation in {time()-self.t0:.3f} sec')
        return predictions

    def __tune_hyperparams(self, dataloaders, nn_structure, partition_id):

        if (self.config.find_lr | self.config.find_structure | self.config.find_l1):
            self.logger.info('# running hyper-parameter tunning ...')

            if self.config.find_lr:
                self.__find_lr(dataloaders['train'], nn_structure, partition_id)
            
            if self.config.find_structure:
                nn_structure = self.__find_structure(dataloaders, nn_structure) # will update nn_structure

            if self.config.find_l1:
                pass
        
        return nn_structure

    def __find_structure(self, dataloaders, nn_structure):
        ''' NN structure tunning
        '''
        self.logger.info('# running nn structure finder ...')

        adamw_kwargs = dict(lr=self.config.learning_rate, **__adamw_kwargs__)
        params = dict(device=self.config.device,
                      nepochs=__nepochs_hyperparams__,
                      adamw_kw=adamw_kwargs,
                      seed=__global_seed__,
                      verbose=False)

        num_features, num_output = nn_structure[2], nn_structure[3]
        structures = [(3, 20, num_features, num_output), 
                      (4, 20, num_features, num_output), 
                      (5, 20, num_features, num_output)]

        loss_fn = self.Loss(**self.config.loss_kwargs)
        best_structure = src.tune_model_structure(self.Model, dataloaders, loss_fn, structures, params)
        self.logger.info(f'found best structure {best_structure} in {time()-self.t0:.3f} sec')
        return best_structure

    def __find_lr(self, train_dataloader, nn_structure, partition_id):
        """
        Find learning rate

        """
        self.logger.info('# running learning rate finder ... ')
        lrfig_path = os.path.join(self.config.output_path, f'loss_vs_lr_{partition_id}.png')

        model = self.Model(*nn_structure)
        optimizer = src.AdamW(params=model.parameters(), lr=1.0e-7, **__adamw_kwargs__)
        loss_fn = self.Loss(**self.config.loss_kwargs)
        lr_finder = src.LRFinder(model, optimizer, loss_fn, device=self.config.device)
        lr_finder.range_test(train_dataloader, end_lr=1, num_iter=300)
        lr_finder.plot(lrfig_path=lrfig_path)
        lr_finder.reset()

        exit(f'LR finder done in {time()-self.t0:.3f} sec, check out {lrfig_path}')
        
    def __run_rfe(self, axes):
        """ Runs Recursive Feature Selection """
        raise NotImplementedError('the checkpoint cannot be loaded due to shape being not saved')
        self.logger.info('# running rfe ...')
        axes_to_keep = {}
        model = src.LinearRegression(add_bias=True)
        num_partitions = 5 if self.config.do_kfold else 1

        for partition_id in range(num_partitions):
            datasets, stats = self.ld.load_data(batch_size=-1, partition_id=partition_id,
                                                loss_fn=self.Loss(**self.config.loss_kwargs))
            fs = src.FeatureElimination(model, datasets)
            fs.run(axes)
            axes_to_keep[partition_id] = fs.results['axes_to_keep'] # we only want the axes
        
        self.logger.info(f'RFE done in {time()-self.t0:.3f} sec')
        for key in axes_to_keep:
            self.logger.info(f"{key}: {axes_to_keep[key]}")            
        return axes_to_keep
 
    def __axes_for_partition(self, partition_id):
        """ Returns axes (indices of imaging maps) for partition 'partition_id' """
        if self.config.do_rfe:
            return self.axes_from_rfe[partition_id]
        else:
            return self.config.axes

    def __plot_losses(self, losses, stats, lossfig_path):
        """ Plots loss vs epochs for training and validation """
        __, train_losses, val_losses = losses
        plt.figure()
        plt.plot(train_losses, 'k-', label='Training')
        plt.plot(val_losses,'r--', label='Validation')
        
        c = ['r', 'k']
        ls = ['--', '-']
        
        for i, baseline in enumerate(['base_val_loss', 'base_train_loss']):
            if baseline in stats:
                plt.axhline(stats[baseline], c=c[i], ls=':', lw=1)
                            
        plt.legend()
        plt.ylabel(self.config.loss.upper()) # MSE or NPLL
        plt.xlabel('Epochs')
        plt.savefig(lossfig_path, bbox_inches='tight')
        plt.close()
        
    # def __find_l1(self, l1_alpha=1.0e-6, seed=42):
    #     if self.config.find_l1:
    #         ''' L1 regularization finder
    #         '''
    #         raise RuntimeError('Not tested')
    #         self.logger.info('L1 regularization scale is being tunned')
    #         model = self.Model(*self.best_structure)
    #         optimizer = src.AdamW(params=model.parameters(), **self.adamw_kw)
    #         criterion = self.Cost(**self.cost_kwargs)
    #         self.l1_alpha = src.tune_L1(model,
    #                             self.dataloaders,
    #                             criterion,
    #                             optimizer,
    #                             10, #self.config.nepochs,
    #                             self.device)
    #         self.logger.info(f'find best L1 scale in {time()-self.t0:.3f} sec')
    #     else:
    #         self.l1_alpha = l1_alpha
    #     self.logger.info(f'l1_alpha: {self.l1_alpha}')
