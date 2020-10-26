SYSNet: an end-to-end imaging systematics cleaning pipeline
===========================================================

[![arXiv](https://img.shields.io/badge/arXiv-1907.11355-b31b1b.svg)](https://arxiv.org/pdf/1907.11355)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SYSNet** is an open source software based on Python for modeling the imaging systematics in large-scale structure data. The observed galaxy density maps are a combination of true signal and systematic noise. The latter is inevitably unknown, however, we do have a set of templates for observational realities that may be the potential sources of systematic error. SYSNet uses a fully connected neural network to model the response to the templates. The methodology is decribed in [Rezaie, Seo, Ross, Bunescu (2020)](https://doi.org/10.1093/mnras/staa1231). The code incorporates:

1. Cyclic Learning Rate (Loshchilov & Hutter 2016 / ICLR 2017)
2. Batch Normalization (Ioffe, Sergey; Szegedy, Christian 2015)
3. Different Cost functions: Mean Squared Error and Poisson Negative log-likelihood.
4. Different Optimizers (e.g., SGD, AdamW, Adam)

Core developers: Mehdi Rezaie, Reza Katebi

Build Status
============
We plan to perform integrated tests of the code, including all models in a miniconda environment for Python 3.8.

[![Build Status](https://travis-ci.org/mehdirezaie/sysnetdev.svg?branch=master)](https://travis-ci.org/mehdirezaie/sysnetdev)


Installation
============
**SYSNet** relies primarily on Pytorch, scikit-learn, numpy, fitsio, and healpy. We recommend using Conda for installation, particularly beacause it allows creating new environments that do not impact the entire computing system. The installation can be divided into three steps:
1. Install/update Conda 
2. Install Pytorch 
3. Install miscellaneous packages 
4. 'Install' SYSNet  

Throughout this note, we use '$>' to denote the commands that ought to be executed in the terminal.

1. Install/Update Conda: First, you need to make sure you have Conda installed or updated in your system. Use the commandd `$> which conda` to see if conda is installed in the system. If not, please follow the instructions below to install conda:  
1.a Visit https://docs.conda.io/projects/conda/en/latest/index.html (or https://docs.conda.io/en/latest/miniconda.html#linux-installers for linux)  
Then, we would execute the following commands on Linux:  
1.b `$> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`  
1.c `$> sha256sum Miniconda3-latest-Linux-x86_64.sh`  



2. Install Pytorch: We recommend to take a look at the [Pytorch website](https://render.githubusercontent.com/view/pytorch.org) to learn more about the framework. The installation of Pytorch on GPU-available machines is different from CPU-only machines. For instance, to set up on the Ohio State Cluster (OSC), you should execute the next two commands to load the CUDA library. For other supercomputers, e.g., NERSC, you may need to read the documentation to see how you can load the CUDA library:  
2.a `$> module spider cuda # on OSC`  
2.b `$> module load cuda/10.1.168 # on OSC`  
For all other devices, i.e., CPU only, you can skip steps 2.a and 2.b, and follow the following steps to create the conda environment (e.g., called sysnet):  
2.c `$> conda create -n sysnet python=3.8 scikit-learn`  
Once your environment is created, you must activate it and use the appropriate Pytorch installation command to install Pytorch. For instance:  
2.d `$> conda activate sysnet`  
2.e `$> conda install pytorch torchvision -c pytorch`  
**Note**: The last step, for the OSC (see https://www.osc.edu/resources/available_software/software_list/cuda & https://www.osc.edu/supercomputing/batch-processing-at-osc/monitoring-and-managing-your-job), will be like:  
2.e `$> conda install pytorch torchvision cudatoolkit=10.1 -c pytorch # on OSC`

3. Install miscellaneous packages: After installation of Pytorch, execute the following commands to install the required packages:  
3.a `$> conda install git jupyter ipykernel ipython mpi4py`  
3.b `$> conda install -c conda-forge fitsio healpy absl-py pytables`  
Use the following command to add your env kernel (e.g., sysnet) to Jupyter:  
3.c `$> python -m ipykernel install --user --name=sysnet --display-name "python (sysnet)"`

4. Install SYSNet: Currently SYSNet is under development and we have not made it pip or conda installable. The only way to set it up is to clone the git repository. You should go to a desired directory (e.g. 'test' under the home directory), and clone the SYSNet repository:  
```
$> cd $HOME
$> mkdir test
$> cd test
$> git clone https://github.com/mehdirezaie/sysnetdev.git 
```
After cloning, you should pull from the master branch to make sure the local repo is updated. To this end, go to the root directory of the sysnet software and pull from origin master:  
```
$> cd sysnetdev
$> git pull origin master
```
Then, insert the absolute path to SYSNET (or _sysnetdev_ directory) to the environment variable PYTHONPATH:  
`$> export PYTHONPATH=/Users/mehdi/test/sysnetdev:${PYTHONPATH}`  
Congratulations! you have successfully set up SYSNet. **Note**: You can add the last line to either ${HOME}/.bashrc or ${HOME}/.bash_profile, so everytime your system reboots, the environment variable PYTHONPATH is set. Also, this is another hack to use the pipeline on-the-go in Python (e.g., in a Jupyter Notebook):  
```Python
import sys
sys.path.insert(0, '/Users/mehdi/test/sysnetdev')
```
In order to test if the installation went correctly. Navigate to the _scripts_ directory, and run the script _app.py_:  
```
$> cd scripts
$> python app.py -ax {0..17}
```
The last command will train the network for one epoch. 

**Note**: Use `$> python app.py --help` to see the help for the full command line interface.

You can also run the test in interactive Python or Jupyter:  
```Python
import sys 
# add the path to SYSNet
sys.path.append('/Users/mehdi/github/sysnetdev') 
from sysnet import SYSNet, Config

# read the default config file which is in the directory 'scripts'
config = Config('config.yaml')                   
pipeline = SYSNet(config)                        
pipeline.run()
```
Please email me at mr095415@ohio.edu if you encounter any issues.

