[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-00ff00.svg)](https://arxiv.org/pdf/1907.11355)

This repository reimplements the SYSNet software in Pytorch. The methodology is decribed in https://doi.org/10.1093/mnras/staa1231. Improvements include:
1. Cyclic Learning Rate (Loshchilov & Hutter 2016 / ICLR 2017)
2. Batch Normalization (Ioffe, Sergey; Szegedy, Christian 2015)

Also, the pipeline now allows different optimizers (e.g., SGD, AdamW) and cost functions (MSE or Poisson log-likelihood)

Contributors: Mehdi Rezaie, Reza Katebi
