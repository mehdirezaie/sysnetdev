
__all__ = ['parse_cmd_arguments']

def parse_cmd_arguments(parser):
    ''' command argument parser
    '''
    parser.add_argument('--input_path',
                        type=str,
                        default='../input/eBOSS.ELG.NGC.DR7.table.fits',
                        help='path to the input data')

    parser.add_argument('--output_path',
                        type=str,
                        default='../checkpoints/model_test.pt',
                        help='path to the output')

    parser.add_argument('--batch_size',
                        type=int,
                        default=4098,
                        help='minibatch size')

    parser.add_argument('--nepochs',
                        type=int,
                        default=11,
                        help='number of training epochs')

    parser.add_argument('--find_lr',
                        default=False,
                        action='store_true',
                        help='find the best learning rate')

    parser.add_argument('--find_structure',
                        default=False,
                        action='store_true',
                        help='find the best nn structure')

    parser.add_argument('--find_l1',
                        default=False,
                        action='store_true',
                        help='find the best L1 lambda')

    return parser.parse_args()
