''' I/O utils
'''
import torch
import fitsio as ft
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader

__all__ = ['LoadData']


class LoadData:

    def __init__(self, fits_file, isKfold=False, random_seed=42):
        self.random_seed = random_seed
        self.isKfold = isKfold
        self.df = ft.read(fits_file) # ('label', 'hpind', 'features', 'fracgood')

        if self.isKfold:
            self.df_kfold = self.split2Kfolds(self.df, k=5, shuffle=True,
                                    random_seed=random_seed)
        else:
            self.df_split = self.split(self.df)  # 5-fold

    def load_data(self, batch_size=1024, partition_id=0):
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

        if self.isKfold:
            assert -1 < partition_id < 5
            train, valid, test = self.df_kfold[partition_id]
        else:
            train, valid, test = self.df_split

        # Z-score normalization
        stats = {
                'x':(np.mean(train['features'], 0), np.std(train['features'], 0)),
                'y':(np.mean(train['label'], 0), np.std(train['label'], 0))
                }

        train = ImagingData(train, stats)
        valid = ImagingData(valid, stats)
        test = ImagingData(test, stats)

        datasets_len = {
                        'train':train.x.shape[0],
                        'valid':valid.x.shape[0],
                        'test':test.x.shape[0],
                        }

        datasets = {
                    'train':MyDataSet(train.x, train.y),
                    'valid':MyDataSet(valid.x, valid.y),
                    'test':MyDataSet(test.x, test.y),
                    }

        dataloaders = {
                        x:DataLoader(datasets[x],
                                    batch_size=batch_size,
                                    shuffle=True)
                                    for x in ['train', 'valid', 'test']
                      }
        return dataloaders, datasets_len, stats

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

    def __init__(self, dt, stats=None):
        self.x = dt['features']
        self.y = dt['label']
        self.p = dt['hpind'].astype('int64')
        self.w = dt['fracgood'].astype('float32')

        if stats is not None:
            self.x = (self.x - stats['x'][0]) / stats['x'][1]
            self.y = (self.y - stats['y'][0]) / stats['y'][1]

        self.x = self.x.astype('float32')
        self.y = self.y.astype('float32')


class MyDataSet(Dataset):

    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).unsqueeze(-1)

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        return (data, label)

    def __len__(self):
        return len(self.x)
