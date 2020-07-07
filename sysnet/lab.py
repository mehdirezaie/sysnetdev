import os
import sys
import logging
from time import time
import numpy as np

import torch
import sysnet.sources as src

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




__all__ = ['SYSNet']
        
        

class SYSNet:
    """
        Implementation of a multilayer neural network for mitigation of
        observational systematics
    """
    logger = logging.getLogger()
    
    def __init__(self, ns):
        """
            Initializes SYSNet
                1. checks the I/O paths
                2. sets the device, i.e., CPU or GPU
                3. sets the loss function, e.g. mse or pll
            
            args:
                ns: namespace, it has all of the parameters
        """
        self.t0 = time()
        self.ns = ns 
        src.check_io(self.ns.input_path, self.ns.output_path) # check I/O        
        src.set_logger(ns.output_path.replace('.pt', '.log'))        
        for (key, value) in ns.__dict__.items():
            self.logger.info(f'{key}: {value}')
            
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check device
        self.logger.info(f'device: {self.device}')
        
        self.Cost, self.cost_kwargs = src.init_loss(self.ns.loss)
        self.collector = src.SYSNetCollector()
        self.Model = src.init_model(self.ns.model)
        
    def load(self):              
        self.ld = src.LoadData(self.ns.input_path, do_kfold=self.ns.do_kfold) # check data
        self.logger.info(f'data loaded in {time()-self.t0:.3f} sec')
             

    def run(self,
           eta_min=1.0e-5,
           lr_best=1.0e-3,
           best_structure=(3, 20, 18, 1),
           l1_alpha=1.0e-3,
           savefig=True,
           seed=42):
        
        if self.ns.do_rfe:
            self.rfe_output = self.__run_rfe(self.ns.axes)
        
        if self.ns.do_kfold:
            output_path_org = self.ns.output_path

            for partition in range(5):
                key = f'partition_{partition}'
                self.ns.output_path = output_path_org.replace('.pt', '_%d.pt'%partition)                
                axes = self.rfe_output[key]['axes_to_keep'] if self.ns.do_rfe else self.ns.axes                
                structure = (best_structure[0], best_structure[1], len(axes), best_structure[3])
                
                self.dataloaders, self.stats = self.ld.load_data(batch_size=self.ns.batch_size,
                                                                 partition_id=partition,
                                                                 normalization=self.ns.normalization,
                                                                 axes=axes,
                                                                 criterion=self.Cost(**self.cost_kwargs))
                
                metrics, hpind, pred = self.__run(eta_min=eta_min,
                                                lr_best=lr_best,
                                                best_structure=structure,
                                                l1_alpha=l1_alpha,
                                                savefig=savefig,
                                                seed=seed)
                
                self.collector.collect(key, {**metrics, **self.stats}, hpind, pred)
            self.ns.output_path = output_path_org

        else:
            key = 'partition_0'
            axes = self.rfe_output[key]['axes_to_keep'] if self.ns.do_rfe else self.ns.axes
            structure = (best_structure[0], best_structure[1], len(axes), best_structure[3])
            
            self.dataloaders, self.stats = self.ld.load_data(batch_size=self.ns.batch_size,
                                                             partition_id=0, # only one partition exists
                                                             normalization=self.ns.normalization,
                                                             axes=axes,
                                                             criterion=self.Cost(**self.cost_kwargs))

            metrics, hpind, pred = self.__run(eta_min=eta_min,
                                            lr_best=lr_best,
                                            best_structure=structure,
                                            l1_alpha=l1_alpha,
                                            savefig=savefig,
                                            seed=seed)
            
            self.collector.collect(key, {**metrics, **self.stats}, hpind, pred)
            
        self.collector.save(self.ns.output_path.replace('.pt', '.json'), self.ns.nside)

        
    def __run_rfe(self, axes):
        axes_to_keep = {}
        model = src.LinearRegression(add_bias=True)
        if self.ns.do_kfold:
            for partition in range(5):
                datasets, stats = self.ld.load_data(batch_size=-1, 
                                                    partition_id=partition,
                                                    criterion=self.Cost(**self.cost_kwargs))
                fs = src.FeatureElimination(model, datasets)
                fs.run(axes)
                axes_to_keep[f'partition_{partition}'] = fs.results
        else:
            datasets, stats = self.ld.load_data(batch_size=-1)
            fs = src.FeatureElimination(model, datasets)
            fs.run(axes)  
            axes_to_keep['partition_0'] = fs.results
        
        self.logger.info(f'RFE done in {time()-self.t0:.3f} sec')
        for key in axes_to_keep:
            self.logger.info(f"{key}: {axes_to_keep[key]['axes_to_keep']}")
            
        return axes_to_keep
 
    def __run(self,
            eta_min=1.0e-5,
            lr_best=1.0e-3,
            best_structure=(3, 20, 18, 1),
            l1_alpha=1.0e-3,
            savefig=True,
            seed=42):
        self.__find_lr(best_structure, eta_min, lr_best)
        self.__find_structure(best_structure)
        self.__find_l1(l1_alpha)
        train_val_losses = self.__train(savefig=savefig, seed=seed)
        test_loss, hpind, pred = self.__evaluate()
        return {**train_val_losses, **test_loss}, hpind, pred

    def __train(self, savefig=True, seed=42):
        ''' TRAINING
        '''
        model = self.Model(*self.best_structure, seed=seed)
        optimizer = src.AdamW(params=model.parameters(), **self.adamw_kw)
        scheduler = src.CosineAnnealingWarmRestarts(optimizer,
                                               T_0=10,
                                               T_mult=2,
                                               eta_min=self.eta_min)
        criterion = self.Cost(**self.cost_kwargs)
        train_losses, val_losses, best_val_loss = src.train_val(model=model,
                                                                dataloaders=self.dataloaders,
                                                                criterion=criterion,
                                                                optimizer=optimizer,
                                                                nepochs=self.ns.nepochs,
                                                                device=self.device,
                                                                output_path=self.ns.output_path,
                                                                scheduler=scheduler,
                                                                L1lambda=self.l1_alpha,
                                                                L1norm=False) # fixme

        self.logger.info(f'finish training in {time()-self.t0:.3f} sec')
        # save train and validation losses
        #np.savez(self.ns.output_path.replace('.pt', '_loss.npz'),
        #        **{'train_losses':train_losses, 'val_losses':val_losses})

        if savefig:
            plt.figure()
            plt.plot(train_losses, 'k-', label='Training')
            plt.plot(val_losses,'r--', label='Validation')
            
            c = ['r', 'k']
            ls = ['--', '-']
            
            for i, baseline in enumerate(['base_val_loss', 'base_train_loss']):
                if baseline in self.stats:
                    plt.axhline(self.stats[baseline], c=c[i], ls=':', lw=1)
                               
            plt.legend()
            plt.ylabel(self.ns.loss.upper()) # MSE or NPLL
            plt.xlabel('Epochs')
            plt.savefig(self.ns.output_path.replace('.pt', '_loss.png'),
                        bbox_inches='tight')
            plt.close()
            self.logger.info(f'make Loss vs epoch plot in {time()-self.t0:.3f} sec')

        return {'min_train_loss':min(train_losses), 
                'min_val_loss':best_val_loss, 
                'train_losses':train_losses, 
                'val_losses':val_losses}


    def __evaluate(self):
        ''' EVALUATE
        '''
        model = self.Model(*self.best_structure)
        model.load_state_dict(torch.load(self.ns.output_path))
        criterion = self.Cost(**self.cost_kwargs)
        test_loss, hpind, pred = src.evaluate(model=model,
                                            dataloaders=self.dataloaders,
                                            criterion=criterion,
                                            device=self.device,
                                            phase='test')
        self.logger.info(f'finish evaluation in {time()-self.t0:.3f} sec')
        self.logger.info(f'test loss: {test_loss:.3f}')
        return {'test_loss':test_loss}, hpind, pred
    

    def __find_lr(self, best_structure=(4, 20, 18, 1), eta_min=1.0e-5, lr_best=1.0e-3, seed=42):

        if self.ns.find_lr:
            ''' LEARNING RATE
            '''
            # --- find learning rate
            fig, ax = plt.subplots()
            model = self.Model(*best_structure)
            optimizer = src.AdamW(params=model.parameters(),
                              lr=1.0e-7,
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=0.01,
                              amsgrad=False)
            criterion = self.Cost(**self.cost_kwargs)
            lr_finder = src.LRFinder(model, 
                                     optimizer,
                                     criterion, 
                                     device=self.device)
            lr_finder.range_test(self.dataloaders['train'],
                                end_lr=1, num_iter=300)
            lr_finder.plot(ax=ax) # to inspect the loss-learning rate graph
            lr_finder.reset()
            fig.savefig(self.ns.output_path.replace('.pt', '_lr.png'),
                        bbox_inches='tight')
            self.logger.info(f'LR finder done in {time()-self.t0:.3f} sec')
            sys.exit()
        else:
            pass # read from the arguments
            self.lr_best = lr_best # manually set these two
            self.eta_min = eta_min
        self.logger.info(f'lr_best: {self.lr_best}')
        self.logger.info(f'eta_min: {self.eta_min}')

        self.adamw_kw = dict(lr=self.lr_best,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=0.01,
                            amsgrad=False)

    def __find_structure(self, best_structure=(4, 20, 18, 1)):

        if self.ns.find_structure:
            ''' NN structure tunning
            '''
            self.logger.info('NN structure is being tunned')
            structures = [(3, 20, 18, 1), (4, 20, 18, 1), (5, 20, 18, 1)]
            criterion = self.Cost(**self.cost_kwargs)
            self.best_structure = src.tune_model_structure(self.Model,
                                                          self.dataloaders,
                                                          criterion,
                                                          10, #self.ns.nepochs,
                                                          self.device,
                                                          structures,
                                                          adamw_kw=self.adamw_kw)

            self.logger.info(f'find best structure in {time()-self.t0:.3f} sec')

        else:
            self.best_structure = best_structure
            
        self.logger.info(f'best_structure: {self.best_structure}')

    def __find_l1(self, l1_alpha=1.0e-6, seed=42):
        if self.ns.find_l1:
            ''' L1 regularization finder
            '''
            self.logger.info('L1 regularization scale is being tunned')
            model = self.Model(*self.best_structure)
            optimizer = src.AdamW(params=model.parameters(), **self.adamw_kw)
            criterion = self.Cost(**self.cost_kwargs)
            self.l1_alpha = src.tune_L1(model,
                                self.dataloaders,
                                criterion,
                                optimizer,
                                10, #self.ns.nepochs,
                                self.device)
            self.logger.info(f'find best L1 scale in {time()-self.t0:.3f} sec')
        else:
            self.l1_alpha = l1_alpha
        self.logger.info(f'l1_alpha: {self.l1_alpha}')
