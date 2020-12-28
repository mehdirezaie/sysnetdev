import sys
sys.path.insert(0, '..')
from sysnet import Config, SYSNet, SYSNetSnapshot

config = Config('config.yaml')
config.update('nepochs', 10)
config.update('snapshot_ensemble', False) # true
config.update('do_kfold', True)

net = SYSNet(config)
net.run()
