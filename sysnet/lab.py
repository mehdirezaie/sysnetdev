"""
    Modeling of Imaging Systematics with multilayer neural networks

    Mehdi Rezaie, mr095415@ohio.edu
    July 17. 2020
"""
import matplotlib.pyplot as plt
import os
import sys
import logging

import numpy as np
import fitsio as ft
from time import time
from glob import glob

import torch
import sysnet.sources as src

import matplotlib
matplotlib.use('Agg')


# set some global variables which do not change
__logger_level__ = 'info'  # info, debug, or warning
__global_seed__ = 85
__nepochs_hyperparams__ = 50
__seed_max__ = 4294967295  # i.e., 2**32 - 1, maximum number in numpy


class SYSNet:
    """
    A multilayer fully connected neural network pipeline for systematics modeling

    TODOs:
        - LOSS for training is different from evaluation (L1 term)
            should we report RMSE only or RMSE + L1?

    methods
    -------
    run : the main method that runs the pipeline


    attributes
    ----------
    config : (argparse.ArgumentParser)
        hyperparameters, some of them are command line arguments:
        ---------------------------
            input_path: (str) path to input file e.g., ../input/eBOSS.ELG.NGC.DR7.table.fits
            output_path: (str) 'root' path to outputs e.g., ../output/model_test
            restore_model: (str) name of the best model e.g., 'best'
            batch_size: (int) batch size, e.g., 4098
            nepochs: (int) number of training epochs
            nchains: (int) number of independent chains with different random init. of weights and biases
            find_lr: (bool) find learning rate
            find_structure: (bool) find best structure
            find_l1: (bool) find L1 scale
            do_kfold: (bool) perform 5-fold validation
            normalization: (str) normalization of features e.g., z-score or minmax
            model: (str) name of the neural network architecture e.g., dnn or dnnp
            axes: (list of int) indices for imaging maps
                e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            do_rfe: (bool) perform recursive feature elimination
            eta_min: (float) minimum learning rate, e.g., 1e-05
            learning_rate: (float) initial learning rate, e.g., 0.02
            nn_structure: (tuple of int), number of layers and units, e.g. (4, 20)
            l1_alpha: (float) L1 regularization scale, e.g., 0.001
            loss: (str) name of the loss function, e.g., mse or pnll

            # Notet that these two attributes are assigned during runtime.
            loss_kwargs: (dict){'reduction': 'sum'}
            device: cpu

    logger : (logging.getLogger)
        logger of the pipeline

    collector : (SYSNetCollector)
        collects the results of different chains and partitions

    Model : (torch.nn.model)
        Object of the neural network architecture

    Loss : (torch.nn.MSELoss or torch.nn.PoissonNLLLoss)
        Object of the loss function


    """
    logger = logging.getLogger()

    def __init__(self, config):
        """
        Initializes SYSNet, i.e., set
            1. logger
            2. loss function
            3. model
            4. data loader
            5. paths to outputs

        parameters
        ----------
        config : (argparse.ArgumentParser)
            see above
        """
        self.t0 = time()
        self.config = config

        log_path = os.path.join(self.config.output_path, 'train.log')
        src.set_logger(log_path, level=__logger_level__)
        self.logger.info(f"logging in {log_path}")

        # initialize loss, collector, model, optimizor
        self.Loss, self.config.loss_kwargs = src.init_loss(self.config.loss)
        self.collector = src.SYSNetCollector()
        self.Model = src.init_model(self.config.model)
        self.Optim, self.config.optim_kwargs = src.init_optim(self.config.optim)
        self.Scheduler, self.config.scheduler_kwargs = src.init_scheduler(self.config)
        
        
        self.config.device = src.get_device() # set the device

        self.logger.info('# --- inputs params ---')
        for (key, value) in self.config.__dict__.items():
            self.logger.info(f'{key}: {value}')
        self.logger.info(f"pipeline initialized in {time()-self.t0:.3f} s")

        self.ld = src.MyDataLoader(self.config.input_path,
                                   do_kfold=self.config.do_kfold,
                                   seed=__global_seed__)
        if self.config.axes == ['all']:
            self.config.axes = np.arange(self.ld.df_split[0]['features'].shape[1])
        else:
            self.config.axes = [int(i) for i in self.config.axes]
        self.logger.info(f'updated axes to {self.config.axes}')
        self.logger.info(f'data loaded in {time()-self.t0:.3f} sec')

        # ---- set the paths
        self.weights_path = os.path.join(
            self.config.output_path, 'nn-weights.fits') # NN output, prediction, used as 'weight' for cosmology
        self.metrics_path = os.path.join(
            self.config.output_path, 'metrics.npz')     # traing and val losses, mean & std of featuers, ..
        self.checkpoint_path_fn = lambda pid, sd: os.path.join(
            self.config.output_path, f'model_{pid}_{sd}')
        self.restore_path_fn = lambda pid, sd: os.path.join(
            self.checkpoint_path_fn(pid, sd), f'best.pth.tar')
        self.lossfig_path_fn = lambda pid, sd: os.path.join(
            self.checkpoint_path_fn(pid, sd), f'loss_model_{pid}_{sd}.png')
        self.lrfig_path_fn = lambda pid: os.path.join(
            self.config.output_path, f'loss_vs_lr_{pid}.png')

    def run(self):
        """
        Run SYSNet with 5-fold validation

        1. run recursive feature selection (optional)
        2. load training, validation, and test sets for given partition
        3. perform hyper-parameter tunning
        4. perform training and evaluation for 'nchain' times with diff. random initializations
            models are saved as .pth.tar
        5. save the metrics (stats of features & label, losses for baseline model and nn) as .npz
            and the predictions as .fits

        """
        self.logger.info('# running pipeline ...')
        if self.config.do_rfe: # to do recursive feature elimination
            self.axes_from_rfe = self.run_rfe(self.config.axes)
            sys.exit('rfe done...rerun the code without --do_rfe')

        self.logger.info('# training and evaluation')
        num_partitions = 5 if self.config.do_kfold else 1

        for partition_id in range(0, num_partitions):  # k-fold validation loop
            axes = self.axes_for_partition(partition_id)
            # if non-linear (# units, # hidden layers, # input layer units, # output unit)
            nn_structure = self.get_structure(len(axes))
            
            self.logger.info(f'partition_{partition_id} with {nn_structure}')

            dataloaders = self.ld.load_data(batch_size=self.config.batch_size,
                                                   partition_id=partition_id,
                                                   normalization=self.config.normalization,
                                                   axes=axes)
            

            self.tune_hyperparams(dataloaders, nn_structure, partition_id)

            self.train_and_eval_chains(dataloaders, nn_structure, partition_id)  # for 'nchains' times

        if not self.config.no_eval:
            self.logger.info(f'wrote weights: {self.weights_path}')
            self.logger.info(f'wrote metrics: {self.metrics_path}')
            self.collector.save(self.weights_path, self.metrics_path)
            if self.config.do_tar:
                self.tar_models(self.config.output_path)

    def get_base_losses(self, dataloaders):
        
        loss_fn = self.Loss(**self.config.loss_kwargs)
        baseline_losses = src.compute_baseline_losses(dataloaders, loss_fn)
        
        for sample, base_loss in baseline_losses.items():
            self.logger.info(f'{sample}: {base_loss:.6f}')
            
        return baseline_losses

    def train_and_eval_chains(self, dataloaders, nn_structure, partition_id):
        """
        Train and evaluate for 'nchain' times

        parameters
        ----------
        dataloaders : dataloaders,
            i.e., 'train', 'test', and 'valid'
        nn_structure : tuple of int
            i.e., (# units, # hidden layers, # input layer units, # output unit)
        partition_id : int

        """
        np.random.seed(__global_seed__)
        seeds = np.random.randint(0, __seed_max__, size=self.config.nchains)

        self.collector.start()
        base_losses = self.get_base_losses(dataloaders)

        for chain_id in range(self.config.nchains):

            seed = seeds[chain_id] + 1000
            self.logger.info(f'# running training and evaluation with seed: {seed}')

            train_val_losses = self.train(dataloaders, nn_structure, seed, partition_id)
            
            if not self.config.no_eval:
                
                restore_path = self.restore_path_fn(partition_id, seed)
                test_loss, hpix, pred_ = self.evaluate(dataloaders['test'], nn_structure, restore_path)
                
                if isinstance(test_loss, list):                
                    self.logger.info(f'best val loss: {train_val_losses[0]:.6f}, (mean) test loss: {np.mean(test_loss):.6f}')
                else:
                    self.logger.info(
                        f'best val loss: {train_val_losses[0]:.6f}, test loss: {test_loss:.6f}')
                    
                self.collector.collect_chain(train_val_losses, test_loss, pred_)

        if not self.config.no_eval:
            self.collector.finish(base_losses, hpix)

    def train(self, dataloaders, nn_structure, seed, partition_id):
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
        """
        checkpoint_path = self.checkpoint_path_fn(partition_id, seed)
        lossfig_path = self.lossfig_path_fn(partition_id, seed)

        model = self.Model(*nn_structure, seed=seed)
        optim_kwargs = dict(lr=self.config.learning_rate, **self.config.optim_kwargs)
        optimizer = self.Optim(params=model.parameters(), **optim_kwargs)
        scheduler = self.Scheduler(optimizer, **self.config.scheduler_kwargs) if self.Scheduler is not None else None
        loss_fn = self.Loss(**self.config.loss_kwargs)
        params = dict(nepochs=self.config.nepochs, device=self.config.device,
                      verbose=True, l1_alpha=self.config.l1_alpha)

        losses = src.train_and_eval(model, optimizer, loss_fn, dataloaders, params,
                                    checkpoint_path=checkpoint_path, scheduler=scheduler,
                                    restore_model=self.config.restore_model, return_losses=True,
                                    snapshot_ensemble=self.config.snapshot_ensemble)
        self.plot_losses(losses, lossfig_path)
        self.logger.info(f'finished training in {time()-self.t0:.3f} sec, checkout {lossfig_path}')
        return losses

    def evaluate(self, dataloader, nn_structure, restore_path):
        """
        Evaluates a trained neural network on the test set

        parameters
        ----------
        dataloader :
        nn_structure : (tuple of int)
            i.e., (# layers, # units, # features, 1)
        restore_path : (str)
            path to the file to restore the weights from


        returns
        -------
        predictions : (test loss, hpix, pred)
            test loss
            healpix pixel indices
            predicted number of galaxies in the pixel

        see also
        --------
        1. https://pytorch.org/tutorials/beginner/saving_loading_models.html
        """
        self.logger.info(f"Restoring parameters from {restore_path}")

        params = dict(device=self.config.device)
        model = self.Model(*nn_structure)
        src.load_checkpoint(restore_path, model)
        model = model.to(params['device'])
        loss_fn = self.Loss(**self.config.loss_kwargs)

        predictions = src.evaluate(model, loss_fn, dataloader, params, return_ypred=True)
        # self.logger.info(f'finish evaluation in {time()-self.t0:.3f} sec')
        return predictions
    
    def get_structure(self, input_dim):
        if self.config.model in ['dnn', 'dnnp', 'dnnps']:
            return (*self.config.nn_structure, input_dim, 1)
        elif self.config.model in ['lin', 'linp']:
            return (input_dim, 1)
        else:
            raise ValueError(f'{self.config.model} is not defined.')


    def tune_hyperparams(self, dataloaders, nn_structure, partition_id):
        """
        Tune hyper-parameters including
            1. learning rate          
            2. number of hidden layers and units
            3. L1 regularization scale


        parameters
        ----------
        dataloaders :
        nn_structure : (tuple of ints)
        partition_id : (int)

        returns
        -------
        nn_structure : tuple of ints
            e.g., (# layers, # units, # features, 1)

        """

        if (self.config.find_lr | self.config.find_structure | self.config.find_l1):
            self.logger.info('# running hyper-parameter tunning ...')

            if self.config.find_lr:
                self.find_lr(dataloaders['train'],
                               nn_structure, partition_id)
                # exit

            if self.config.find_structure:
                self.find_structure(dataloaders, nn_structure)  
                # exit

            if self.config.find_l1:
                self.find_l1alpha(dataloaders, nn_structure)
                # exit

    def find_structure(self, dataloaders, nn_structure):
        """
        Tune NN structure by trying
            1 hidden layers x 20 units
            4 hidden layers x 20 units
            5 hidden layers x 20 units

        parameters
        ----------
        dataloaders :
        nn_structure : tuple of int


        returns
        -------
        best_structure : tuple of int
        """
        self.logger.info('# running nn structure finder ...')

        optim_kwargs = dict(lr=self.config.learning_rate, **self.config.optim_kwargs)
        params = dict(device=self.config.device,
                      nepochs=__nepochs_hyperparams__,
                      optim_kw=optim_kwargs,
                      seed=__global_seed__,
                      verbose=True,
                      l1_alpha=self.config.l1_alpha)

        num_features, num_output = nn_structure[2], nn_structure[3]
        structures = [(3, 20, num_features, num_output),
                      (4, 20, num_features, num_output),
                      (5, 20, num_features, num_output),
                      (6, 20, num_features, num_output)]

        loss_fn = self.Loss(**self.config.loss_kwargs)
        best_structure = src.tune_model_structure(self.Model, self.Optim, dataloaders, loss_fn, structures, params)
        self.logger.info(f'found best structure {best_structure} in {time()-self.t0:.3f} sec')
        sys.exit()

    def find_lr(self, train_dataloader, nn_structure, partition_id):
        """
        Find learning rate



        see also
        --------
        1. Cyclical Learning Rates for Training Neural Networks, https://arxiv.org/abs/1506.01186
        2. fastai/lr_find: https://github.com/fastai/fastai
        """
        self.logger.info('# running learning rate finder ... ')
        lrfig_path = self.lrfig_path_fn(partition_id)
        
        model = self.Model(*nn_structure)
        optimizer = self.Optim(params=model.parameters(),
                              lr=1.0e-7, **self.config.optim_kwargs)
        loss_fn = self.Loss(**self.config.loss_kwargs)
        lr_finder = src.LRFinder(model, optimizer, loss_fn, device=self.config.device)
        lr_finder.range_test(train_dataloader, end_lr=1, num_iter=300)
        lr_finder.plot(lrfig_path=lrfig_path)
        lr_finder.reset()

        sys.exit(f'LR finder done in {time()-self.t0:.3f} sec, check out {lrfig_path}')

    def run_rfe(self, axes):
        """ Runs Recursive Feature Selection """
        self.logger.info('# running rfe ...')
        axes_to_keep = {}
        model = src.LinearRegression(add_bias=True)
        num_partitions = 5 if self.config.do_kfold else 1

        for partition_id in range(num_partitions):
            datasets = self.ld.load_data(batch_size=-1, partition_id=partition_id)
            fs = src.FeatureElimination(model, datasets)
            fs.run(axes)
            # we only want the axes
            axes_to_keep[partition_id] = fs.results['axes_to_keep']

        self.logger.info(f'RFE done in {time()-self.t0:.3f} sec')
        for key in axes_to_keep:
            self.logger.info(f"{key}: {axes_to_keep[key]}")
        return axes_to_keep

    def axes_for_partition(self, partition_id):
        """ Returns axes (indices of imaging maps) for partition 'partition_id' """
        if self.config.do_rfe:
            return self.axes_from_rfe[partition_id]
        else:
            return self.config.axes

    def plot_losses(self, losses, lossfig_path):
        """ Plots loss vs epochs for training and validation """
        __, train_losses, val_losses = losses
        plt.figure()
        plt.plot(train_losses, 'k-', label='Training', lw=1)
        plt.plot(val_losses, 'r--', label='Validation')

        c = ['r', 'k']
        ls = ['--', '-']

        #for i, baseline in enumerate(['base_valid_loss', 'base_train_loss']):
        #    if baseline in baseline_losses:
        #        plt.axhline(baseline_losses[baseline], c=c[i], ls=':', lw=1)

        plt.legend()
        plt.ylabel(self.config.loss.upper())  # MSE or NPLL
        plt.xlabel('Epochs')
        plt.savefig(lossfig_path, bbox_inches='tight')
        plt.close()

    def find_l1alpha(self, dataloaders, nn_structure):
        """
        parameters
        ----------
        dataloaders :
        nn_structure : tuple of int


        returns
        -------
        best_structure : tuple of int
        """
        self.logger.info('# running L1 alpha finder ...')
        optim_kwargs = dict(lr=self.config.learning_rate, **self.config.optim_kwargs)
        params = dict(device=self.config.device,
                      nepochs=__nepochs_hyperparams__,
                      optim_kw=optim_kwargs,
                      seed=__global_seed__,
                      verbose=False,
                      l1_alpha=self.config.l1_alpha)

        l1_alphas = np.logspace(-8, 1, 10)

        model = self.Model(*nn_structure, seed=params['seed'])
        loss_fn = self.Loss(**self.config.loss_kwargs)
        best_l1_alpha = src.tune_l1_scale(model, self.Optim, dataloaders, loss_fn, l1_alphas, params)
        
        self.logger.info(f'found best l1_alpha {best_l1_alpha} in {time()-self.t0:.3f} sec')
        sys.exit()

    def tar_models(self, path_models, model_fmt='model_*_*', tarfile_name='models.tar.gz'):
        """ Tar all models to reduce the number of outputs """
        src.tar_models(path_models, model_fmt='model_*_*', tarfile_name='models.tar.gz')

        
class SYSNetSnapshot(SYSNet):
    def __init__(self, *arrays, **kwargs):
        super(SYSNetSnapshot, self).__init__(*arrays, **kwargs)
    
    def evaluate(self, dataloader, nn_structure, restore_path):
        """
        Evaluates an ensemble trained neural network on the test set

        parameters
        ----------
        dataloader :
        nn_structure : (tuple of int)
            i.e., (# layers, # units, # features, 1)
        restore_path : (str)
            path to the file to restore the weights from


        returns
        -------
        predictions : (test loss, hpix, pred)
            test loss
            healpix pixel indices
            predicted number of galaxies in the pixel

        see also
        --------
        1. https://pytorch.org/tutorials/beginner/saving_loading_models.html
        """
        snapshot_path = os.path.dirname(restore_path)
        snapshots = glob(os.path.join(snapshot_path, 'snapshot_*.pth.tar'))
        
        self.logger.info(f"Restoring parameters from {len(snapshots)} snapshots")
        if len(snapshots) == 0:sys.exit(f'Something is wrong. there is no snapshots.')
        
        pred_ensemble = []
        testloss_ensemble = []
        for snapshot_i in snapshots:
            test_loss_, hpix_, pred_ = super(SYSNetSnapshot, self).evaluate(dataloader, nn_structure, snapshot_i)
            pred_ensemble.append(pred_)
            testloss_ensemble.append(test_loss_)
            
        return testloss_ensemble, hpix_, torch.cat(pred_ensemble, 1)


class TrainedModel:
    
    def __init__(self, model, checkpoint, nnstruct=(4, 20), num_features=17):
        
        self.DNNx = src.init_model(model)
        self.dnnx = self.DNNx(*nnstruct, input_dim=num_features)
        checkpoint = src.load_checkpoint(checkpoint, self.dnnx)
        self.stats =  checkpoint['stats']
        
    def forward(self, indata, axes):
        dl = src.load_data(indata, self.stats, axes)
        
        result = src.forward(self.dnnx, dl, {'device':'cpu'})
        hpix = result[0].numpy()            
        nnw = result[1].numpy().flatten()
        #return Table([hpix, nnw], names=['hpix', 'weight'])
        return (hpix, nnw)
