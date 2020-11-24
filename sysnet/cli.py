from argparse import ArgumentParser
from sysnet.sources.io import Config


def parse_cmd_arguments(yaml_config=None):    
    ''' command argument parser
    '''
    cf = Config(yaml_config)
    ap = ArgumentParser()

    ap.add_argument('-i', '--input_path',
                        type=str,
                        default=cf.fetch('input_path', '../input/eBOSS.ELG.NGC.DR7.table.fits'),
                        help='path to the input data')

    ap.add_argument('-o', '--output_path',
                        type=str,
                        default=cf.fetch('output_path', '../output/model_test'),
                        help='path to the output')

    ap.add_argument('--restore_model',
                        type=str,
                        default=cf.fetch('restore_model', None),
                        help='model to restore the weights from')

    ap.add_argument('-bs', '--batch_size',
                        type=int,
                        default=cf.fetch('batch_size', 512),
                        help='minibatch size')

    ap.add_argument('-ne', '--nepochs',
                        type=int,
                        default=cf.fetch('nepochs', 1),
                        help='number of training epochs')

    ap.add_argument('-nc', '--nchains',
                        type=int,
                        default=cf.fetch('nchains', 1),
                        help='number of chains with different initializations')

    ap.add_argument('-fl', '--find_lr',
                        default=cf.fetch('find_lr', False),
                        action='store_true',
                        help='find the best learning rate')

    ap.add_argument('-fs', '--find_structure',
                        default=cf.fetch('find_structure', False),
                        action='store_true',
                        help='find the best nn structure')

    ap.add_argument('-fl1', '--find_l1',
                        default=cf.fetch('find_l1', False),
                        action='store_true',
                        help='find the best L1 alpha')

    ap.add_argument('-k', '--do_kfold',
                        default=cf.fetch('do_kfold', False),
                        action='store_true',
                        help='enable k-fold cross validation (k=5)')

    ap.add_argument('--do_tar',
                        default=cf.fetch('do_tar', False),
                        action='store_true',
                        help='tar all the models')

    ap.add_argument('-norm', '--normalization',
                        type=str,
                        default=cf.fetch('normalization', 'z-score'), 
                        help='standardization method')

    ap.add_argument('--model',
                        type=str,
                        default=cf.fetch('model', 'dnnp'),
                        help='model, dnn or dnnp')

    ap.add_argument('--optim',
                        type=str,
                        default=cf.fetch('optim', 'adamw'),
                        help='adamw or sgd')

    ap.add_argument('--scheduler',
                        type=str,
                        default=cf.fetch('scheduler', 'cosann'),
                        help='cosann or none')

    ap.add_argument('-ax', '--axes',
                        type=int,
                        nargs='*',
                        default=cf.fetch('axes', [0]),
                        #required=True,
                        help='index of features to use')

    ap.add_argument('--do_rfe',
                        default=cf.fetch('do_rfe', False),
                        action='store_true',
                        help='perform recursive feature elimination')

    ap.add_argument('--eta_min',
                        type=float,
                        default=cf.fetch('eta_min', 1.0e-5),
                        help='min eta for LR finder')

    ap.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=cf.fetch('learning_rate', 1.0e-3),
                        help='best initial learning rate')

    ap.add_argument('--nn_structure',
                        type=int,
                        default=cf.fetch('nn_structure', (4, 20)),
                        nargs='*',
                        help='structure ( # hidden layer, # neurons)')

    ap.add_argument('--l1_alpha',
                        type=float,
                        default=cf.fetch('l1_alpha', -1.0),
                        help='L1 scale (negative value will turn off regularization)')

    ap.add_argument('--loss',
                        type=str,
                        default=cf.fetch('loss', 'pnll'),
                        help='Cost function (loss) e.g., mse, pnll')

    return ap.parse_args()
