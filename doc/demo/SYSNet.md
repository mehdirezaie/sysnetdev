# SYSNet: Systematics Treatment with Neural Networks

`SYSNet` was developed to tackle the problem of imaging systematic effects, e.g., Galactic dust, in galaxy survey data.

## Installation
We recommend Conda for installation:
```bash
  conda create -q -n sysnet python=3.8 scikit-learn git jupyter ipykernel ipython mpi4py matplotlib
  conda activate sysnet
  conda install pytorch torchvision -c pytorch
  conda install -c conda-forge fitsio healpy absl-py pytables pyyaml
```

## Summary

#### Preprocessing
The input data to `SYSNet` is a tabulated data of the following fields:
1. `hpix`: (int) HEALPix index
2. `label`: (float) number count of galaxies (or quasars) in pixel
3. `fracgood`: (float) weight associated to pixel (the network output will be multiplied by this factor)
4. `features`: (float) array holding imaging properties for pixel

#### Neural Network regression
The `SYSNet` software is called in this step to perform a regression analysis modeling the relationship between `label` and `features`.

#### Postprocessing
The `SYSNet` output will be used to assign appropriate _weights_ to galaxies to account for observational systematics.

## Demo
In the following, we aim to provide a demo of the regression and postprocessing steps.


```python
%matplotlib inline
import matplotlib.pyplot as plt
import sys 
import numpy as np
import healpy as hp
import fitsio as ft
sys.path.append('/home/mehdi/github/sysnetdev') # 'Cause SYSNet is not installed yet
from sysnet import SYSNet, Config, TrainedModel
%matplotlib inline
```

### Input data


```python
input_ = ft.read('../input/eBOSS.ELG.NGC.DR7.table.fits')  # read tab. data

for array in input_.dtype.descr:
    print(array)
     
# for visualization    
nside = 256
ng = np.zeros(12*nside*nside)
ng[:] = np.nan
ng[input_['hpix']] = input_['label']    

# Mollweide projection
hp.mollview(ng, rot=-85, min=0.5, max=2.0,
            title='Observed galaxy density [normalized]')
```

    ('label', '>f8')
    ('hpix', '>i8')
    ('features', '>f8', (18,))
    ('fracgood', '>f8')



![png](output_5_1.png)



```python
config = Config('../scripts/config.yaml')   # read config file
```


```python
config.__dict__
```




    {'input_path': '../input/eBOSS.ELG.NGC.DR7.table.fits',
     'output_path': '../output/model_test',
     'restore_model': None,
     'batch_size': 512,
     'nepochs': 2,
     'nchains': 1,
     'find_lr': False,
     'find_structure': False,
     'find_l1': False,
     'do_kfold': False,
     'do_tar': False,
     'snapshot_ensemble': False,
     'normalization': 'z-score',
     'model': 'dnn',
     'optim': 'adamw',
     'scheduler': 'cosann',
     'axes': [0, 1, 2],
     'do_rfe': False,
     'eta_min': 1e-05,
     'learning_rate': 0.001,
     'nn_structure': [4, 20],
     'l1_alpha': -1.0,
     'loss': 'mse'}



Here is what each keywork means:
```
'input_path': '../input/eBOSS.ELG.NGC.DR7.table.fits',    # path to the input file
 'output_path': '../output/model_test',                   # dir path to the outputs
 'restore_model': None,                                   # if you want to resume training?
 'batch_size': 512,                                       # size of the mini-batch for training
 'nepochs': 2,                                            # max number of training epochs
 'nchains': 1,                                            # number of chains for the ensemble
 'find_lr': False,                                        # run learning rate finder
 'find_structure': False,                                 # run nn structure finder (brute force)
 'find_l1': False,                                        # run L1 scale finder (brute force)
 'do_kfold': False,                                       # perfom k-fold validation (k=5)
 'do_tar': False,                                         # tar some of the output files 
 'snapshot_ensemble': False,                              # run snapshot ensemble, train one get M for free
 'normalization': 'z-score',                              # normalization rule for the features
 'model': 'dnn',                                          # name of the model, e.g., dnn or dnnp
 'optim': 'adamw',                                        # name of the optimizer
 'scheduler': 'cosann',                                   # name of the scheduler
 'axes': [0, 1, 2],                                       # list of feature indices
 'do_rfe': False,                                         # perform recursive feature elimination (ablation)
 'eta_min': 1e-05,                                        # min learning rate
 'learning_rate': 0.001,                                  # initial learning rate
 'nn_structure': [4, 20],                                 # structure of the neural net (# layers, # units)     
 'l1_alpha': -1.0,                                        # L1 scale, if < 0, it will be ignored
 'loss': 'mse'                                            # name of the loss function, e.g., mse or pnll
```

### Important hyper-parameter, The learning rate!


```python
# let's update some input arguments
config.update('nepochs', 30)
config.update('axes', [i for i in range(18)]) # num of features for this dataset
config.update('batch_size', 4096)

# run learning rate finder
config.update('find_lr', True)
pipeline = SYSNet(config) # perform regression
pipeline.run()
```

    logging in ../output/model_test/train.log
    # --- inputs params ---
    input_path: ../input/eBOSS.ELG.NGC.DR7.table.fits
    output_path: ../output/model_test
    restore_model: None
    batch_size: 4096
    nepochs: 30
    nchains: 1
    find_lr: True
    find_structure: False
    find_l1: False
    do_kfold: False
    do_tar: False
    snapshot_ensemble: False
    normalization: z-score
    model: dnn
    optim: adamw
    scheduler: cosann
    axes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    do_rfe: False
    eta_min: 1e-05
    learning_rate: 0.001
    nn_structure: [4, 20]
    l1_alpha: -1.0
    loss: mse
    loss_kwargs: {'reduction': 'sum'}
    optim_kwargs: {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0, 'amsgrad': False}
    scheduler_kwargs: {'eta_min': 1e-05, 'T_0': 10, 'T_mult': 2}
    device: cpu
    pipeline initialized in 0.042 s
    data loaded in 0.329 sec
    # running pipeline ...
    # training and evaluation
    partition_0 with (4, 20, 18, 1)
    base_train_loss: 0.140
    base_valid_loss: 0.142
    base_test_loss: 0.144
    # running hyper-parameter tunning ...
    # running learning rate finder ... 


    Learning rate search finished. See the graph with {finder_name}.plot()



    An exception has occurred, use %tb to see the full traceback.


    SystemExit: LR finder done in 100.810 sec, check out ../output/model_test/loss_vs_lr_0.png



    /home/mehdi/miniconda3/envs/sysnet/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3339: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
      warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)


Check out the produce plot:
![this image](../output/model_test/loss_vs_lr_0.png)

Based on this plot, we should use the initial learning rate around 0.01.


```python
# let's re-train the network with the best learning rate
# monitor validation loss

config.update('find_lr', False) # remember to turn this off
config.update('learning_rate', 0.01)
pipeline = SYSNet(config) # perform regression
pipeline.run()
```

    logging in ../output/model_test/train.log
    # --- inputs params ---
    input_path: ../input/eBOSS.ELG.NGC.DR7.table.fits
    output_path: ../output/model_test
    restore_model: None
    batch_size: 4096
    nepochs: 30
    nchains: 1
    find_lr: False
    find_structure: False
    find_l1: False
    do_kfold: False
    do_tar: False
    snapshot_ensemble: False
    normalization: z-score
    model: dnn
    optim: adamw
    scheduler: cosann
    axes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    do_rfe: False
    eta_min: 1e-05
    learning_rate: 0.01
    nn_structure: [4, 20]
    l1_alpha: -1.0
    loss: mse
    loss_kwargs: {'reduction': 'sum'}
    optim_kwargs: {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0, 'amsgrad': False}
    scheduler_kwargs: {'eta_min': 1e-05, 'T_0': 10, 'T_mult': 2}
    device: cpu
    pipeline initialized in 0.019 s
    data loaded in 0.148 sec
    # running pipeline ...
    # training and evaluation
    partition_0 with (4, 20, 18, 1)
    base_train_loss: 0.140
    base_valid_loss: 0.142
    base_test_loss: 0.144
    # running training and evaluation with seed: 2664485226
    Epoch 0/29 train loss: 0.541742 valid loss: 0.182060 lr: 0.009773
    Epoch 1/29 train loss: 0.153096 valid loss: 0.141687 lr: 0.009079
    Epoch 2/29 train loss: 0.135798 valid loss: 0.137821 lr: 0.007986
    Epoch 3/29 train loss: 0.134558 valid loss: 0.137206 lr: 0.006602
    Epoch 4/29 train loss: 0.133914 valid loss: 0.136802 lr: 0.005061
    Epoch 5/29 train loss: 0.133466 valid loss: 0.136629 lr: 0.003515
    Epoch 6/29 train loss: 0.133110 valid loss: 0.136457 lr: 0.002115
    Epoch 7/29 train loss: 0.132818 valid loss: 0.136348 lr: 0.000997
    Epoch 8/29 train loss: 0.132662 valid loss: 0.136214 lr: 0.000272
    Epoch 9/29 train loss: 0.132546 valid loss: 0.136207 lr: 0.000010
    Epoch 10/29 train loss: 0.133582 valid loss: 0.137068 lr: 0.009943
    Epoch 11/29 train loss: 0.133404 valid loss: 0.136004 lr: 0.009764
    Epoch 12/29 train loss: 0.132805 valid loss: 0.135629 lr: 0.009468
    Epoch 13/29 train loss: 0.132020 valid loss: 0.135185 lr: 0.009062
    Epoch 14/29 train loss: 0.131359 valid loss: 0.134681 lr: 0.008557
    Epoch 15/29 train loss: 0.130826 valid loss: 0.134388 lr: 0.007964
    Epoch 16/29 train loss: 0.130392 valid loss: 0.134042 lr: 0.007298
    Epoch 17/29 train loss: 0.129999 valid loss: 0.133840 lr: 0.006575
    Epoch 18/29 train loss: 0.129673 valid loss: 0.133601 lr: 0.005814
    Epoch 19/29 train loss: 0.129405 valid loss: 0.133310 lr: 0.005033
    Epoch 20/29 train loss: 0.129155 valid loss: 0.133169 lr: 0.004251
    Epoch 21/29 train loss: 0.128935 valid loss: 0.133199 lr: 0.003488
    Epoch 22/29 train loss: 0.128662 valid loss: 0.133028 lr: 0.002762
    Epoch 23/29 train loss: 0.128372 valid loss: 0.132881 lr: 0.002092
    Epoch 24/29 train loss: 0.128161 valid loss: 0.132745 lr: 0.001493
    Epoch 25/29 train loss: 0.127999 valid loss: 0.132738 lr: 0.000980
    Epoch 26/29 train loss: 0.127876 valid loss: 0.132718 lr: 0.000567
    Epoch 27/29 train loss: 0.127798 valid loss: 0.132609 lr: 0.000263
    Epoch 28/29 train loss: 0.127710 valid loss: 0.132605 lr: 0.000076
    Epoch 29/29 train loss: 0.127680 valid loss: 0.132601 lr: 0.000010
    finished training in 298.419 sec, checkout ../output/model_test/model_0_2664485226/loss_model_0_2664485226.png
    Restoring parameters from ../output/model_test/model_0_2664485226/best.pth.tar
    best val loss: 0.132601, test loss: 0.133700
    wrote weights: ../output/model_test/nn-weights.fits
    wrote metrics: ../output/model_test/metrics.npz



![png](output_13_1.png)


This plot shows the training and validation loss vs epoch.
![this image](../output/model_test/model_0_2664485226/loss_model_0_2664485226.png)

The baseline model returns 0.144 for the test loss, while the neural network is able to yield a lower value, 0.130.The validation loss is also not showing any sign of over-fitting. 

The code outputs several files:
1. `nn-weights.fits`: this file has healpix index and predicted galaxy count
2. `metrics.npz`: training, validation, test loss and mean and std of features
3. `best.pth.tar`: the best model parameters


```python
tm = TrainedModel('dnn', # name of the model
                  '../output/model_test/model_0_2664485226/best.pth.tar', # best model part 0 seed 2664485226
                  0, # partition of the model
                  (4, 20), # structure of the network
                  18) # num of input features

hpix, npred = tm.forward('../output/model_test/metrics.npz',  # metrics, we need the mean and std of features
                         '../input/eBOSS.ELG.NGC.DR7.table.fits') 
npred = npred / npred.mean()  # normalize
npred = npred.clip(0.5, 2.0)  # avoid extreme predictions


ng_ = np.zeros(12*256*256)
ng_[:] = np.nan
ng_[hpix] = npred
```


```python
hp.mollview(ng, rot=-85, min=0.5, max=2.0,
            title='Observed galaxy density [normalized]')

hp.mollview(ng_, rot=-85, min=0.5, max=2.0,
            title='Predicted galaxy density [normalized]')
```


![png](output_17_0.png)



![png](output_17_1.png)


For 30 training epochs, a single fold, and a single chain model, this is not a bad result. In practice, we use 5-fold validation (`do_kfold=True`) and train 20 chains (`nchains=20`) for70-300 epochs.

If you have any questions, feel free to email me at mr095415@ohio.edu.
