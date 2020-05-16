
__all__ = ['parse_cmd_arguments']

def parse_cmd_arguments(parser):
    ''' command argument parser
    '''
    parser.add_argument('-i', '--input_path',
                        type=str,
                        default='../input/eBOSS.ELG.NGC.DR7.table.fits',
                        help='path to the input data')

    parser.add_argument('-o', '--output_path',
                        type=str,
                        default='../checkpoints/model_test.pt',
                        help='path to the output')

    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        default=4098,
                        help='minibatch size')

    parser.add_argument('-ne', '--nepochs',
                        type=int,
                        default=1,
                        help='number of training epochs')

    parser.add_argument('-fl', '--find_lr',
                        default=False,
                        action='store_true',
                        help='find the best learning rate')

    parser.add_argument('-fs', '--find_structure',
                        default=False,
                        action='store_true',
                        help='find the best nn structure')

    parser.add_argument('-fl1', '--find_l1',
                        default=False,
                        action='store_true',
                        help='find the best L1 lambda')

    parser.add_argument('-k', '--isKfold',
                        default=False,
                        action='store_true',
                        help='enable k-fold cross validation (k=5)')

    parser.add_argument('-norm', '--normalization',
                        type=str,
                        default='z-score',
                        help='standardization method')

    return parser.parse_args()
