''' I/O utils
'''
import os
import logging
import json
from json import JSONEncoder
import yaml

import torch
from torch.utils.data import Dataset, DataLoader

import fitsio as ft
import numpy as np
import healpy as hp

from sklearn.model_selection import train_test_split, KFold


def tohp(nside, hpix, values):
    zeros = np.empty(12*nside*nside, dtype=values.dtype)
    zeros[:] = np.nan
    zeros[hpix] = values
    return zeros

class Config:
    def __init__(self, config_file):
        # https://stackoverflow.com/a/1639197/9746916
        self.config = read_config_yml(config_file)
        for k, v in self.config.items():
            setattr(self, k, v)

    def fetch(self, key, default):
        try:
            tmp = self.config[key]
        except:
            tmp = default
        return tmp

def read_config_yml(path_yml):
    with open(path_yml, 'r') as f:
        conf = yaml.safe_load(f.read())
        return conf

def save_checkpoint(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'best.pth.tar'.
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'best.pth.tar')
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    torch.save(state, filepath)


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        scheduler: 
    """
    if not os.path.exists(checkpoint):
        raise RuntimeError(f"File doesn't exist {checkpoint}")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_dict'])

    return checkpoint


class SYSNetCollector:
    """
    Collects results

    """

    def __init__(self):
        self.stats = {}
        self.losses = {'train': [],
                       'valid': [],
                       'test': []
                       }
        self.pred = []
        self.hpix = []
        self.key = 0

    def start(self):
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.pred_list = []

    def collect_chain(self, train_val_losses, test_loss, pred_):
        self.train_losses.append(train_val_losses[1])
        self.valid_losses.append(train_val_losses[2])
        self.test_losses.append(test_loss)
        self.pred_list.append(pred_)

    def finish(self, stats, hpix):
        self.stats[self.key] = stats
        self.hpix.append(hpix)
        self.losses['train'].append(self.train_losses)
        self.losses['valid'].append(self.valid_losses)
        self.losses['test'].append(self.test_losses)
        self.pred.append(torch.cat(self.pred_list, 1))
        self.key += 1

    def save(self, weights_path, metrics_path, nside=None):
        """ save metrics and predictions """
        weights_dir = os.path.dirname(weights_path)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        metrics_dir = os.path.dirname(metrics_path)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        pred = torch.cat(self.pred, 0).numpy()
        hpix = torch.cat(self.hpix, 0).numpy()
        weights = np.zeros(pred.shape[0], dtype=[
                           ('hpix', 'i8'), ('weight', 'f8', (pred.shape[1], ))])
        weights['hpix'] = hpix
        weights['weight'] = pred

        ft.write(weights_path, weights, clobber=True)
        np.savez(metrics_path, stats=self.stats, losses=self.losses)

        # --- MR: how about we save as json? e.g.,
        # with open(output_path, 'w') as output_file:
        #     json.dump({'hpix':self.hpix,
        #                 'pred':self.pred,
        #                 'metrics':self.metrics},
        #                 output_file, cls=NumpyArrayEncoder)

        # hpmap = tohp(nside, self.hpix, self.pred)
        # hp.write_map(output_path.replace('.json', f'.hp{nside}.fits'), hpmap,
        #             overwrite=True, dtype=hpmap.dtype)


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

    # if not output_path.endswith('.pt'):
    #    raise ValueError(f'{output_path} must end with .pt')

    output_dir = os.path.dirname(output_path)   # check output dir
    if not os.path.exists(output_dir):
        # fixme: create a dir
        raise RuntimeError(f'{output_dir} does not exist')


class MyDataLoader:

    logger = logging.getLogger()

    def __init__(self, input_file, do_kfold=False, seed=42):
        self.seed = seed
        self.do_kfold = do_kfold

        if input_file.endswith('.fits'):
            self.df_split = self.__read_fits(input_file)
        elif input_file.endswith('.npy'):
            self.df_split = self.__read_npy(input_file)

    def load_data(self, batch_size=1024,
                  partition_id=0, normalization='z-score',
                  add_bias=False, axes=None):
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

        # df = ft.read(fits_file)   # ('label', 'hpix', 'features', 'fracgood')

        if self.do_kfold:
            assert -1 < partition_id < 5
            train, valid, test = self.df_split[partition_id]
        else:
            train, valid, test = self.df_split

        if normalization == 'z-score':
            # Z-score normalization
            stats = {
                'x': (np.mean(train['features'], 0), np.std(train['features'], 0)),
                'y': (np.mean(train['label'], 0), np.std(train['label'], 0))
            }
        elif normalization == 'minmax':
            # min-max
            stats = {
                'x': (np.min(train['features'], 0),
                      np.max(train['features'], 0)-np.min(train['features'], 0)),
                'y': (np.min(train['label'], 0),
                      np.max(train['label'], 0)-np.min(train['label'], 0)),
            }
        else:
            raise NotImplementedError(f'{normalization} not implemented')

        train = ImagingData(train, stats, add_bias=add_bias, axes=axes)
        valid = ImagingData(valid, stats, add_bias=add_bias, axes=axes)
        test = ImagingData(test, stats, add_bias=add_bias, axes=axes)

        datasets = {
            'train': MyDataSet(train.x, train.y, train.p, train.w),
            'valid': MyDataSet(valid.x, valid.y, valid.p, valid.w),
            'test': MyDataSet(test.x, test.y, test.p, test.w),
        }

        if batch_size==-1:
            return datasets, stats
        else:
            dataloaders = {
                s: DataLoader(datasets[s],
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
                for s in ['train', 'valid', 'test']
            }
            return dataloaders, stats

    def __read_npy(self, npy_file):
        ''' old npy file i.e., already split into 5 folds
        '''
        df_raw = np.load(npy_file, allow_pickle=True).item()
        df = {}
        for i in range(5):
            df[i] = (df_raw['train']['fold%d' % i],
                     df_raw['validation']['fold%d' % i],
                     df_raw['test']['fold%d' % i])
        if self.do_kfold:
            return df
        else:
            return df[0]

    def __read_fits(self, fits_file):
        # ('label', 'hpix', 'features', 'fracgood')
        self.df = ft.read(fits_file)
        if self.do_kfold:
            return self.__split2Kfolds(self.df,
                                       k=5,
                                       shuffle=True,
                                       seed=self.seed)
        else:
            return self.__split(self.df, seed=self.seed)  # 5-fold

    def __split2Kfolds(self, data, k=5, shuffle=True, seed=42):
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
        np.random.seed(seed)
        kfold = KFold(k, shuffle=shuffle, random_state=seed)
        index = np.arange(data.size)
        kfold_data = {
            0: {},
            1: {},
            2: {},
            3: {},
            4: {}
        }
        for i, (nontestID, testID) in enumerate(kfold.split(index)):
            #
            #
            validID = np.random.choice(
                nontestID, size=testID.size, replace=False)
            trainID = np.setdiff1d(nontestID, validID)
            kfold_data[i] = (data[trainID], data[validID], data[testID])
            
        return kfold_data

    def __split(self, df, seed=42):
        train, test = train_test_split(df, test_size=0.2, random_state=seed)
        train, valid = train_test_split(
            train, test_size=0.25, random_state=seed)
        return train, valid, test


class ImagingData(object):
    """ 
    - currently scales features only
    """

    def __init__(self, dt, stats=None, add_bias=False, axes=None):
        self.x = dt['features']
        self.y = dt['label']
        try:
            self.p = dt['hpix'].astype('int64')
        except:
            self.p = dt['hpind'].astype('int64')

        self.w = dt['fracgood'].astype('float32')

        if stats is not None:
            assert np.all(stats['x'][1] >
                          0), 'feature with 0 variance detected!'
            self.x = (self.x - stats['x'][0]) / stats['x'][1]
            # self.y = (self.y - stats['y'][0]) / stats['y'][1] # don't scale label

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
        self.w = torch.from_numpy(w).unsqueeze(-1)

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        hpix = self.p[index]
        weight = self.w[index]
        return (data, label, weight, hpix)

    def __len__(self):
        return len(self.x)
