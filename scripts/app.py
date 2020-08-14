#!/usr/bin/env python
"""
    TODO:
        - Check the ratio of the training loss to the baseline loss for the MSE and Poisson cost func.
        - Save Hyper-parameters in case we don't want to do hp training again?
        - Add a yaml file to use for the inputs, useful for scaling to 1k mocks
        - with feature selection the shape of the input layer is different, the model cannot be restored
"""
import sysnet

sysnet.test_torch()

config = sysnet.parse_cmd_arguments('config.yaml')
pipeline = sysnet.SYSNet(config)
pipeline.run()