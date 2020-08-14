from sysnet.sources.io import Config
__all__ = ['parse_cmd_arguments']

def parse_cmd_arguments(parser, yaml_config=None):    
    ''' command argument parser
    '''
    if yaml_config is not None:
        cf = Config(yaml_config)
        
    parser.add_argument('-i', '--input_path',
                        type=str,
                        default=cf.fetch('input_path', '../input/eBOSS.ELG.NGC.DR7.table.fits'),
                        help='path to the input data')

    parser.add_argument('-o', '--output_path',
                        type=str,
                        default=cf.fetch('output_path', '../output/model_test'),
                        help='path to the output')

    parser.add_argument('--restore_model',
                        type=str,
                        default=cf.fetch('restore_model', None),
                        help='model to restore the weights from')

    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        default=cf.fetch('batch_size', 4098),
                        help='minibatch size')

    parser.add_argument('-ne', '--nepochs',
                        type=int,
                        default=cf.fetch('nepochs', 1),
                        help='number of training epochs')

    parser.add_argument('-nc', '--nchains',
                        type=int,
                        default=cf.fetch('nchains', 1),
                        help='number of chains with different initializations')

    parser.add_argument('-fl', '--find_lr',
                        default=cf.fetch('find_lr', False),
                        action='store_true',
                        help='find the best learning rate')

    parser.add_argument('-fs', '--find_structure',
                        default=cf.fetch('find_structure', False),
                        action='store_true',
                        help='find the best nn structure')

    parser.add_argument('-fl1', '--find_l1',
                        default=cf.fetch('find_l1', False),
                        action='store_true',
                        help='find the best L1 alpha')

    parser.add_argument('-k', '--do_kfold',
                        default=cf.fetch('do_kfold', False),
                        action='store_true',
                        help='enable k-fold cross validation (k=5)')

    parser.add_argument('-norm', '--normalization',
                        type=str,
                        default=cf.fetch('normalization', 'z-score'), 
                        help='standardization method')

    parser.add_argument('--model',
                        type=str,
                        default=cf.fetch('model', 'dnnp'),
                        help='model, dnn or dnnp')
    
    parser.add_argument('--optim',
                        type=str,
                        default=cf.fetch('optim', 'adamw'),
                        help='adamw or sgd')

    parser.add_argument('-ax', '--axes',
                        type=int,
                        nargs='*',
                        default=cf.fetch('axes', [0]),
                        #required=True,
                        help='index of features to use')

    parser.add_argument('--do_rfe',
                        default=cf.fetch('do_rfe', False),
                        action='store_true',
                        help='perform recursive feature elimination')

    parser.add_argument('--eta_min',
                        type=float,
                        default=cf.fetch('eta_min', 1.0e-5),
                        help='min eta for LR finder')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=cf.fetch('learning_rate', 1.0e-3),
                        help='best initial learning rate')

    parser.add_argument('--nn_structure',
                        type=int,
                        default=cf.fetch('nn_structure', (4, 20)),
                        nargs='*',
                        help='structure ( # hidden layer, # neurons)')

    parser.add_argument('--l1_alpha',
                        type=float,
                        default=cf.fetch('l1_alpha', -1.0),
                        help='L1 scale (negative value will turn off regularization)')

    parser.add_argument('--loss',
                        type=str,
                        default=cf.fetch('loss', 'pnll'),
                        help='Cost function (loss) e.g., mse, pnll')

    return parser.parse_args()
