""" 
    Modeling of Imaging Systematics with multilayer neural networks

    Mehdi Rezaie, mr095415@ohio.edu
    July 17. 2020
"""
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
__logger_level__ = 'info' # info, debug, or warning
__global_seed__ = 85        
__adamw_kwargs__ = dict(betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
__cosann_warmup_kwargs__ = dict(T_0=10, T_mult=2)
__nepochs_hyperparams__ = 2
__seed_max__ = 4294967295 # i.e., 2**32 - 1, maximum number in numpy

class SYSNet:
    """
    A multilayer fully connected neural network pipeline for systematics modeling


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

        # ---- set the paths
        self.weights_path = os.path.join(self.config.output_path, 'nn-weights.fits')
        self.metrics_path = os.path.join(self.config.output_path, 'metrics.npz')
        self.checkpoint_path_fn = lambda pid, sd: os.path.join(self.config.output_path, f'model_{pid}_{sd}') 
        self.restore_path_fn = lambda pid, sd: os.path.join(self.checkpoint_path_fn(pid, sd), f'best.pth.tar')
        self.lossfig_path_fn = lambda pid, sd: os.path.join(self.checkpoint_path_fn(pid, sd), f'loss_model_{pid}_{sd}.png')
        self.lrfig_path_fn = lambda pid: os.path.join(self.config.output_path, f'loss_vs_lr_{partition_id}.png')

             
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

        self.collector.save(self.weights_path, self.metrics_path)

    def __train_and_eval_chains(self, dataloaders, nn_structure, partition_id, stats):
        """ 
        Train and evaluate for 'nchain' times 
        
        parameters
        ----------
        dataloaders : dataloaders, 
            i.e., 'train', 'test', and 'valid'
        nn_structure : tuple of int 
            i.e., (# units, # hidden layers, # input layer units, # output unit)
        partition_id : int

        stats : dict
            stats of features and label, and losses of baseline model

        """
        np.random.seed(__global_seed__)
        seeds = np.random.randint(0, __seed_max__, size=self.config.nchains) # initialization seed        
        
        self.collector.start()

        for chain_id in range(self.config.nchains): 

            seed = seeds[chain_id]
            self.logger.info(f'# running training and evaluation with seed: {seed}')

            train_val_losses = self.__train(dataloaders, nn_structure, seed, partition_id, stats)
        
            restore_path = self.restore_path_fn(partition_id, seed)
            test_loss, hpix, pred_ = self.__evaluate(dataloaders['test'], nn_structure, restore_path)
            self.logger.info(f'best val loss: {train_val_losses[0]:.3f}, test loss: {test_loss:.3f}')
            self.collector.collect_chain(train_val_losses, test_loss, pred_)

        self.collector.finish(stats, hpix)

    def __train(self, dataloaders, nn_structure, seed, partition_id, stats):
        """ 
        Train and evaluate a nn on training and validation sets 
        
        
        parameters
        ----------
        dataloaders : 
        nn_structure : (tuple of int)
        seed : (int)
        partition_id : (int)
        stats : (dict)

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
        #self.logger.info(f'finish evaluation in {time()-self.t0:.3f} sec')
        return predictions

    def __tune_hyperparams(self, dataloaders, nn_structure, partition_id):
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
                self.__find_lr(dataloaders['train'], nn_structure, partition_id)
            
            if self.config.find_structure:
                nn_structure = self.__find_structure(dataloaders, nn_structure) # will update nn_structure

            if self.config.find_l1:
                pass
        
        return nn_structure

    def __find_structure(self, dataloaders, nn_structure):
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



        see also
        --------
        1. Cyclical Learning Rates for Training Neural Networks, https://arxiv.org/abs/1506.01186
        2. fastai/lr_find: https://github.com/fastai/fastai
        """
        self.logger.info('# running learning rate finder ... ')
        lrfig_path = self.lrfig_path_fn(partition_id) 

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
