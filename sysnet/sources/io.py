''' I/O utils
'''
import os
import torch
import fitsio as ft
import numpy as np
import healpy as hp
import logging
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader

import json
from json import JSONEncoder


__all__ = ['LoadData', 'check_io', 'SYSNetCollector']


def tohp(nside, hpind, values):
    zeros = np.empty(12*nside*nside, dtype=values.dtype)
    zeros[:] = np.nan 
    zeros[hpind] = values
    return zeros

class SYSNetCollector:
    
    def __init__(self):
        self.metrics = {}
        self.pred = []
        self.hpind = []
        
    def collect(self, key, metrics, hpind, pred):
        self.metrics[key] = metrics
        self.hpind.append(hpind)
        self.pred.append(pred)
                
    def save(self, output_path, nside):
        self.pred = torch.cat(self.pred).numpy()
        self.hpind = torch.cat(self.hpind).numpy()            
        
        with open(output_path, 'w') as output_file:
            json.dump({'hpind':self.hpind, 
                       'pred':self.pred, 
                       'metrics':self.metrics}, 
                      output_file, cls=NumpyArrayEncoder)
            
        hpmap = tohp(nside, self.hpind, self.pred)
        hp.write_map(output_path.replace('.json', f'.hp{nside}.fits'), hpmap, 
                     overwrite=True, dtype=hpmap.dtype)

class NumpyArrayEncoder(JSONEncoder):
    # https://pynative.com/python-serialize-numpy-ndarray-into-json/
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    

def check_io(input_path, output_path):
    """ checks the paths to input and output
    
    args:
        input_path: str, path to the input file
        output_path: str, path to the output file 
    
    """
    # check input and output
    if not os.path.exists(input_path):
        raise RuntimeError(f'{input_path} does not exist!')

    if os.path.exists(output_path):
        raise RuntimeError(f'{output_path} already exists!')

    if not output_path.endswith('.pt'):
        raise ValueError(f'{output_path} must end with .pt')
    output_dir = os.path.dirname(output_path)   # check output dir
    if not os.path.exists(output_dir):
        raise RuntimeError(f'{output_dir} does not exist') # fixme: create a dir



class LoadData:
    
    logger = logging.getLogger()
    def __init__(self, input_file, do_kfold=False, random_seed=42):
        self.random_seed = random_seed
        self.do_kfold = do_kfold
        
        if input_file.endswith('.fits'):            
            self.df_split = self.read_fits(input_file)
        elif input_file.endswith('.npy'):
            self.df_split = self.read_npy(input_file)
            
                
    def read_npy(self, npy_file):
        ''' old npy file
        '''
        df_raw = np.load(npy_file, allow_pickle=True).item()
        df = {}
        for i in range(5):
            df[i] = (df_raw['train']['fold%d'%i], 
                     df_raw['validation']['fold%d'%i], 
                     df_raw['test']['fold%d'%i])
        if self.do_kfold:
            return df
        else:
            return df[0]
    
    def read_fits(self, fits_file):
        self.df = ft.read(fits_file) # ('label', 'hpind', 'features', 'fracgood')
        if self.do_kfold:
            return self.split2Kfolds(self.df, 
                                     k=5, 
                                     shuffle=True,
                                     random_seed=self.random_seed)
        else:
            return self.split(self.df)  # 5-fold     
        
        

    def load_data(self, batch_size=1024,
                  partition_id=0, normalization='z-score',
                  add_bias=False, axes=None, criterion=None):
        '''
        This function loads the data generators from a fits file.

        inputs
        --------
        fits_file: str, path to fits file
        batch_size: int, batch size, default=1024

        returns
        --------
        dataloaders: dict, holds the generators for training, ... examples
        datasets_len: dict, holds the number of training,... examples
        stats: dict, holds the mean and std of training x and y
        '''
        # read data
        # split into 5 folds
        # five partitions (test, training, validation)
        # partition ID -> test, train, val.

        #df = ft.read(fits_file)   # ('label', 'hpind', 'features', 'fracgood')

        if self.do_kfold:
            assert -1 < partition_id < 5
            train, valid, test = self.df_split[partition_id]
        else:
            train, valid, test = self.df_split
        
        if normalization == 'z-score':
            # Z-score normalization
            stats = {
                    'x':(np.mean(train['features'], 0), np.std(train['features'], 0)),
                    'y':(np.mean(train['label'], 0), np.std(train['label'], 0))
                    }
        elif normalization == 'minmax':
            # min-max
            stats = {
                    'x':(np.min(train['features'], 0),
                         np.max(train['features'], 0)-np.min(train['features'], 0)),
                    'y':(np.min(train['label'], 0),
                         np.max(train['label'], 0)-np.min(train['label'], 0)),
                    }
        else:
            raise NotImplementedError(f'{normalization} not implemented')


        train = ImagingData(train, stats, add_bias=add_bias, axes=axes)
        valid = ImagingData(valid, stats, add_bias=add_bias, axes=axes)
        test = ImagingData(test, stats, add_bias=add_bias, axes=axes)
        
        if criterion is not None:        
            train_ymean = torch.from_numpy(train.y).mean()
            # eq: np.var(train.y) if MSE
            
            baseline_losses = {'base_train_loss':criterion(train_ymean.expand(train.y.size), 
                                                               torch.from_numpy(train.y)).item(),
                               'base_val_loss':criterion(train_ymean.expand(valid.y.size), 
                                                               torch.from_numpy(valid.y)).item(),
                               'base_test_loss':criterion(train_ymean.expand(test.y.size), 
                                                              torch.from_numpy(test.y)).item()}
            for s in baseline_losses:
                self.logger.info(f'{s}: {baseline_losses[s]:.3f}') 
                
            stats = {**stats, **baseline_losses}
            

        datasets = {
                    'train':MyDataSet(train.x, train.y, train.p, train.w),
                    'valid':MyDataSet(valid.x, valid.y, valid.p, valid.w),
                    'test':MyDataSet(test.x, test.y, test.p, test.w),
                    }
        
        if batch_size == -1:
            return datasets, stats
        else:            
            dataloaders = {
                            s:DataLoader(datasets[s],
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0)
                                        for s in ['train', 'valid', 'test']
                          }
            return dataloaders, stats
        

    def split2Kfolds(self, data, k=5, shuffle=True, random_seed=42):
        '''
            split data into k randomly chosen regions
            for training, validation and testing


            data
            |__ 0
            |   |__ train
            |   |__ validation
            |   |__ test
            |
            |__ 1
        '''
        np.random.seed(random_seed)
        kfold = KFold(k, shuffle=shuffle, random_state=random_seed)
        index = np.arange(data.size)
        kfold_data = {
                      0:{},
                      1:{},
                      2:{},
                      3:{},
                      4:{}
                      }
        for i, (nontestID, testID) in enumerate(kfold.split(index)):
            #
            #
            validID  = np.random.choice(nontestID, size=testID.size, replace=False)
            trainID  = np.setdiff1d(nontestID, validID)
            #
            #
            #kfold_data[i]['test'] = data[testID]
            #kfold_data[i]['train'] = data[trainID]
            #kfold_data[i]['validation'] = data[validID]
            kfold_data[i] = (data[trainID], data[validID], data[testID])
        return kfold_data

    def split(self, df, seed=42):
        train, test = train_test_split(df, test_size=0.2, random_state=seed)
        train, valid = train_test_split(train, test_size=0.25, random_state=seed)
        return train, valid, test

class ImagingData(object):

    def __init__(self, dt, stats=None, add_bias=False, axes=None):
        self.x = dt['features']
        self.y = dt['label']
        self.p = dt['hpind'].astype('int64')
        self.w = dt['fracgood'].astype('float32')

        if stats is not None:
            self.x = (self.x - stats['x'][0]) / stats['x'][1]
            #self.y = (self.y - stats['y'][0]) / stats['y'][1]
            
        if axes is not None:
            self.x = self.x[:, axes]

        if add_bias:
            self.x = np.column_stack([np.ones(self.x.shape[0]), self.x])            
        self.x = self.x.astype('float32')
        self.y = self.y.astype('float32')


class MyDataSet(Dataset):

    def __init__(self, x, y, p, w):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).unsqueeze(-1)
        self.p = p
        self.w = w
        
    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        hpind = self.p[index]
        weight = self.w[index]
        return (data, label, weight, hpind)

    def __len__(self):
        return len(self.x)
