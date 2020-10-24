[![arXiv](https://img.shields.io/badge/arXiv-1907.11355-b31b1b.svg)](https://arxiv.org/pdf/1907.11355)
[![License: MIT](https://img.shields.io/badge/License-MIT-006905.svg)](https://opensource.org/licenses/MIT)

This repository reimplements the SYSNet software in Pytorch. The methodology is decribed in https://doi.org/10.1093/mnras/staa1231. Improvements include:
1. Cyclic Learning Rate (Loshchilov & Hutter 2016 / ICLR 2017)
2. Batch Normalization (Ioffe, Sergey; Szegedy, Christian 2015)

Also, the pipeline now allows different optimizers (e.g., SGD, AdamW) and cost functions (MSE or Poisson log-likelihood)

Contributors: Mehdi Rezaie, Reza Katebi

Build Status
============
[![Build Status](https://travis-ci.org/mehdirezaie/sysnetdev.svg?branch=master)](https://travis-ci.org/mehdirezaie/sysnetdev)
