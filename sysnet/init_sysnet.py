import sys
from time import time
import numpy as np

import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sysnet.sources as src

__all__ = ['SYSNet']

class SYSNet:

    def __init__(self, ns):
        self.t0 = time()
        ''' DATA '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device: {self.device}')

        data = src.load_data(ns.input_path, ns.batch_size)
        self.dataloaders, self.datasets_len, self.stats = data
        print(f'data loaded in {time()-self.t0:.3f} sec')
        self.ns = ns

    def run(self,
            eta_min=1.0e-5,
            lr_best=1.0e-3,
            best_structure=(3, 20, 18, 1),
            l1_alpha=1.0e-6,
            savefig=True):

        self.__find_lr(eta_min, lr_best)
        self.__find_structure(best_structure)
        self.__find_l1(l1_alpha)
        self.__train(savefig=savefig)
        self.__evaluate()


    def __find_lr(self, eta_min=1.0e-5, lr_best=1.0e-3):

        if self.ns.find_lr:
            ''' LEARNING RATE
            '''
            # --- find learning rate
            fig, ax = plt.subplots()
            model = src.DNN(3, 20, 18, 1)
            optimizer = AdamW(params=model.parameters(),
                              lr=1.0e-7,
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=0.01,
                              amsgrad=False)
            criterion = MSELoss() # reduction='mean'
            lr_finder = src.LRFinder(model, optimizer,
                                    criterion, device=self.device)
            lr_finder.range_test(self.dataloaders['train'],
                                end_lr=1, num_iter=300)
            lr_finder.plot(ax=ax) # to inspect the loss-learning rate graph
            lr_finder.reset()
            fig.savefig(self.ns.output_path.replace('.pt', '_lr.png'),
                        bbox_inches='tight')
            print(f'LR finder done in {time()-self.t0:.3f} sec')
            sys.exit()
        else:
            pass # read from the arguments
            self.lr_best = lr_best # manually set these two
            self.eta_min = eta_min
        print(f'lr_best: {self.lr_best}')
        print(f'eta_min: {self.eta_min}')

        self.adamw_kw = dict(lr=self.lr_best,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=0.01,
                            amsgrad=False)

    def __find_structure(self, best_structure=(5, 20, 18, 1)):

        if self.ns.find_structure:
            ''' NN structure tunning
            '''
            print('NN structure is being tunned')
            structures = [(3, 20, 18, 1), (4, 20, 18, 1), (5, 20, 18, 1)]
            criterion = MSELoss() # reduction='mean'
            self.best_structure = src.tune_model_structure(src.DNN,
                                                          self.dataloaders,
                                                          self.datasets_len,
                                                          criterion,
                                                          self.ns.nepochs,
                                                          self.device,
                                                          structures,
                                                          adamw_kw=self.adamw_kw)

            print(f'find best structure in {time()-self.t0:.3f} sec')

        else:
            self.best_structure = best_structure
        print(f'best_structure: {self.best_structure}')

    def __find_l1(self, l1_alpha=1.0e-6):
        if self.ns.find_l1:
            ''' L1 regularization finder
            '''
            print('L1 regularization scale is being tunned')
            model = src.DNN(*self.best_structure)
            optimizer = AdamW(params=model.parameters(), **self.adamw_kw)
            criterion = MSELoss() # reduction='mean'
            self.l1_alpha = src.tune_L1(model,
                                self.dataloaders,
                                self.datasets_len,
                                criterion,
                                optimizer,
                                self.ns.nepochs,
                                self.device)
            print(f'find best L1 scale in {time()-self.t0:.3f} sec')
        else:
            self.l1_alpha = l1_alpha
        print(f'l1_alpha: {self.l1_alpha}')

    def __train(self, savefig=True):
        ''' TRAINING
        '''
        model = src.DNN(*self.best_structure)
        optimizer = AdamW(params=model.parameters(), **self.adamw_kw)
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                               T_0=10,
                                               T_mult=2,
                                               eta_min=self.eta_min)
        criterion = MSELoss() # reduction='mean'
        train_losses, val_losses,_ = src.train_val(model=model,
                                            dataloaders=self.dataloaders,
                                            datasets_len=self.datasets_len,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            nepochs=self.ns.nepochs,
                                            device=self.device,
                                            output_path=self.ns.output_path,
                                            scheduler=scheduler)

        print(f'finish training in {time()-self.t0:.3f} sec')
        # save train and validation losses
        np.savez(self.ns.output_path.replace('.pt', '_loss.npz'),
                **{'train_losses':train_losses, 'val_losses':val_losses})

        if savefig:
            plt.figure()
            plt.plot(train_losses, 'b-',
                     val_losses,'b--')
            plt.ylabel('MSE')
            plt.xlabel('Epochs')
            plt.savefig(self.ns.output_path.replace('.pt', '_loss.png'),
                        bbox_inches='tight')
            plt.close()
            print(f'make Loss vs epoch plot in {time()-self.t0:.3f} sec')

    def __evaluate(self):
        ''' EVALUATE
        '''
        model = src.DNN(*self.best_structure)
        model.load_state_dict(torch.load(self.ns.output_path))
        criterion = MSELoss() # reduction='mean'
        test_loss = src.evaluate(model=model,
                            dataloaders=self.dataloaders,
                            datasets_len=self.datasets_len,
                            criterion=criterion,
                            device=self.device,
                            phase='test')
        print(f'finish evaluation in {time()-self.t0:.3f} sec')
        print(f'test loss: {test_loss:.3f}')
