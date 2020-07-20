
__all__ = ['parse_cmd_arguments']


def parse_cmd_arguments(parser):
    ''' command argument parser
    '''
    # parser.add_argument('-d', '--data',
    #                     type=str,
    #                     default='../input/eBOSS_QSO_full_NGC_v7_2.dat.fits',
    #                     help='path to the input data catalog')

    # parser.add_argument('-r', '--randoms',
    #                     type=str,
    #                     default='../input/eBOSS_QSO_full_NGC_v7_2.ran.fits',
    #                     help='path to the input random catalogs')

    # parser.add_argument('-t', '--templates',
    #                     type=str,
    #                     default='../input/SDSS_WISE_HI_imageprop_nside512.h5',
    #                     help='path to the input random catalogs')

    parser.add_argument('-i', '--input_path',
                        type=str,
                        default='../input/eBOSS.ELG.NGC.DR7.table.fits',
                        help='path to the input data')

    parser.add_argument('-o', '--output_path',
                        type=str,
                        default='../output/model_test',
                        help='path to the output')

    parser.add_argument('--restore_model',
                        type=str,
                        default=None,
                        help='model to restore the weights from')

    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        default=4098,
                        help='minibatch size')

    parser.add_argument('-ne', '--nepochs',
                        type=int,
                        default=1,
                        help='number of training epochs')

    parser.add_argument('-nc', '--nchains',
                        type=int,
                        default=1,
                        help='number of chains with different initializations')

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
                        help='find the best L1 alpha')

    parser.add_argument('-k', '--do_kfold',
                        default=False,
                        action='store_true',
                        help='enable k-fold cross validation (k=5)')

    parser.add_argument('-norm', '--normalization',
                        type=str,
                        default='z-score',
                        help='standardization method')

    parser.add_argument('--model',
                        type=str,
                        default='dnn',
                        help='model, dnn or dnnp')

    parser.add_argument('-ax', '--axes',
                        type=int,
                        nargs='*',
                        required=True,
                        help='index of features to use')

    parser.add_argument('--do_rfe',
                        default=False,
                        action='store_true',
                        help='perform recursive feature elimination')

    parser.add_argument('--eta_min',
                        type=float,
                        default=1.0e-5,
                        help='min eta for LR finder')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=1.0e-3,
                        help='best initial learning rate')

    parser.add_argument('--nn_structure',
                        type=int,
                        default=(4, 20),
                        nargs='*',
                        help='structure ( # hidden layer, # neurons)')

    parser.add_argument('--l1_alpha',
                        type=float,
                        default=1.0e-3,
                        help='L1 scale')

    # parser.add_argument('-z', '--zbins',
    #                     type=float,
    #                     nargs='*',
    #                     default=[0.8, 2.2, 3.5],
    #                     help='redshift bin edges')

    # parser.add_argument('--nside',
    #                     type=int,
    #                     default=512,
    #                     help='HEALPix nside')

    parser.add_argument('--loss',
                        type=str,
                        default='mse',
                        help='Cost function (loss) e.g., mse, pnll')

    return parser.parse_args()
