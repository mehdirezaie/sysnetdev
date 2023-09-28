SYSNet: an end-to-end imaging systematics cleaning pipeline
===========================================================

[![arXiv](https://img.shields.io/badge/arXiv-1907.11355-b31b1b.svg)](https://arxiv.org/pdf/1907.11355)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SYSNet** is an open source software based on Python for modeling the imaging systematics in large-scale structure data. The observed galaxy density maps are a combination of true signal and systematic noise. The latter is inevitably unknown, however, we do have a set of templates for observational realities that may be the potential sources of systematic error. SYSNet uses a fully connected neural network to model the response to the templates. The methodology is decribed in [Rezaie, Seo, Ross, Bunescu (2020)](https://doi.org/10.1093/mnras/staa1231). The code incorporates:

1. Cyclic Learning Rate (Loshchilov & Hutter 2016 / ICLR 2017)
2. Batch Normalization (Ioffe, Sergey; Szegedy, Christian 2015)
3. Different Cost functions: Mean Squared Error and Poisson Negative log-likelihood.
4. Different Optimizers (e.g., SGD, AdamW, Adam)

Relevant Literature
===================
Here are the papers that would help understanding this code. Read them in this order:               
1. Sections 1-3, 4.1, 4.6, 4.9, 4.10 from https://arxiv.org/pdf/1609.04747.pdf               
2. Sections 2-3.2 from https://arxiv.org/pdf/1907.11355.pdf (if this is hard to follow, read this Medium article first: https://purnasaigudikandula.medium.com/a-beginner-intro-to-neural-networks-543267bda3c8)               
3. Sections 1-3 from https://arxiv.org/pdf/1608.03983.pdf               
4. Sections 1-3 from https://arxiv.org/pdf/1506.01186.pdf               

If you have time and or are interested in learning more (order doesn't matter):   
a. https://www.nature.com/articles/nature14539               
b. https://arxiv.org/pdf/1502.03167.pdf               
c. https://arxiv.org/pdf/1704.00109.pdf               


Quick Setup
===========
There are two ways.
1. execute `python -m pip install git+https://github.com/mehdirezaie/sysnetdev.git` from your terminal (Thanks to Arnaud De-Mattia).

2. First clone the repository with `git clone https://github.com/mehdirezaie/sysnetdev.git`, and then add the (absolute) path to PYTHONPATH. For instance,
```bash
export PYTHONPATH=/Users/rezaie/github/sysnetdev:$PYTHONPATH
```

Demo
=====
Detailed: For installation and a demo of SYSNet, check out [this demo](https://nbviewer.jupyter.org/github/mehdirezaie/sysnetdev/blob/master/notebooks/demo_decalsdr7.ipynb).


Build Status
============
We plan to perform integrated tests of the code, including all models in a miniconda environment for Python 3.8.

[![Build Status](https://travis-ci.org/mehdirezaie/sysnetdev.svg?branch=master)](https://travis-ci.org/mehdirezaie/sysnetdev)


Please email me at mr095415@ohio.edu if you encounter any issues.

Core developers: Mehdi Rezaie, Reza Katebi
