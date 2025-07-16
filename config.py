from optparse import OptionParser


def get_args():
    parser = OptionParser()

    # parameters for generating test_data records

    parser.add_option('--model_dir', dest='model_dir', type='str',
                      default='',
                      help='directory of model')

    parser.add_option('--raw_data_dir', dest='raw_data_dir', type='str',
                      default='/raid/snac/hualei/brain_data/brain_mask/preprocessed_BET_3d+ms_100',
                      help='directory of raw test_data')
    parser.add_option('--output_data_dir', dest='output_data_dir', type='str',
                      default='/raid/snac/hualei/brain_data/brain_2k+100_2class_hdf5_patch_32_t1_MNI',
                      help='top folder contains training, testing and validation test_data')

    parser.add_option('--train_data_dir', dest='train_data_dir', type='str',
                      default='/media/sggq/Disk/syy2/datasets/old_brain/brain_2class_hdf5_t1_MNI/train',
                      help='directory of training test_data')
    parser.add_option('--val_data_dir', dest='val_data_dir', type='str',
                      default='/media/sggq/Disk/syy/datasets/old_brain/brain_2class_hdf5_t1_MNI/val',
                      help='directory of validation test_data')
    parser.add_option('--test_data_dir', dest='test_data_dir', type='str',
                      default='/raid/snac/hualei/brain_data/brain_2k+100_2class_hdf5_patch_32_t1_MNI/test',
                      help='directory of test test_data')
    parser.add_option('--test_label_dir', dest='test_label_dir', type='str',
                      default='/raid/snac/hualei/brain_data/brain_2k+100_2class_hdf5_patch_32_t1_MNI/test_label',
                      help='directory of test test_data')
    # parameters for test_data I/O
    parser.add_option('-b', '--batch_size', dest='batch_size',
                      type='int', default=4, help='batch size')
    parser.add_option('-p', '--patch_size', dest='patch_size',
                      type='int', default=64, help='patch size')
    parser.add_option('--cv', dest='cv',
                      type='int', default=0, help='5 fold')

    parser.add_option('--log_dir', dest='log_dir',
                      type='str', default="", help='log_dir')
    parser.add_option('--KFold_dir', dest='KFold_dir',
                      type='str', default="", help='KFold_dir')
    parser.add_option('-o', '--overlap_step', dest='overlap_step',
                      type='int', default=8, help='overlap step')
    parser.add_option('-w', '--worker_num', dest='worker_num',
                      type='int', default=8,
                      help='number of workers of dataloader')

    # parameters for training
    parser.add_option('-e', '--epochs', dest='epochs',
                      default=10, type='int',
                      help='number of epochs')
    parser.add_option('-l', '--lr', dest='lr', default=2e-5,
                      type='float', help='learning rate')
    parser.add_option('--resume', dest='resume', default=True,
                      help='resume to train, from checkpoint')
    parser.add_option('--result_dir', dest='result_dir',
                      default='./result-UNeXt3D_IBSR_1000',
                      type='str',
                      help='directory to save model')
    parser.add_option('-c', '--class_num', dest='class_num',
                      type='int', default=2, help='number of class')

    # parameters for test
    parser.add_option('--test_id', dest='test_instance_id',
                      type='int', default=453,
                      help='id of test instance')
    parser.add_option('--checkpoint_num', type='int',
                      default=570000,
                      help='which checkpoint is used for validation / prediction')
    parser.add_option('--epochs_per_vali', dest='epochs_per_vali',
                      type='int', default=50,
                      help='number of training epochs to run before validation')

    # train / test mode switch
    parser.add_option('-m', '--mode', type='str',
                      default='train',
                      help='train or predict')
    # model name choose
    parser.add_option('--model_name', type='str',
                      default='UNeXt3D',
                      help='choose train / test model')
    # random seed
    parser.add_option('--random_seed', type='int',
                      default=5210,
                      help='choose the number you like')
    parser.add_option('--pred_sample',
                      default=False,
                      help='Where need pred some samples could set this parameter to True')

    # GPU
    parser.add_option('-g', '--gpus', type='str',
                      default='0',
                      help='GPU IDs')

    #registion parameters
    # parser.add_option('--img-list', required=True, help='line-seperated list of training files')  # 训练数据文件
    # parser.add_option('--img-prefix', help='optional input image file prefix')  # 文件前缀
    # parser.add_option('--img-suffix', help='optional input image file suffix')  # 文件后缀
    # parser.add_option('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
    # parser.add_option('--flirtReg_dir', help='image after registration by flirt')
    # parser.add_option('--model-dir_reg', default='models',
    #                     help='model output directory (default: models)')
    parser.add_option('--csv_dir', help='Loss of registration')
    # parser.add_option('--multichannel', action='store_true',
    #                     help='specify that data has multiple channels')

    # training parameters
    # parser.add_option('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    # parser.add_option('--batch-size', type=int, default=1, help='batch size (default: 1)')
    # parser.add_option('--epochs', type=int, default=1500,
    #                     help='number of training epochs (default: 1500)')
    # parser.add_option('--steps-per-epoch', type=int, default=100,
    #                     help='frequency of model saves (default: 100)')
    # parser.add_option('--load-model', help='optional model file to initialize with')  # 是否加载模型
    # parser.add_option('--initial-epoch', type=int, default=0,
    #                     help='initial epoch number (default: 0)')
    # parser.add_option('--lr_reg', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    # parser.add_option('--cudnn-nondet', action='store_true',
    #                     help='disable cudnn determinism - might slow down training')

    # network architecture parameters
    # parser.add_option('--enc', type=int, nargs='+',
    #                     help='list of unet encoder filters (default: 16 32 32 32)')  # unet architecture
    # parser.add_option('--dec', type=int, nargs='+',
    #                     help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    # parser.add_option('--int-steps', type=int, default=7,
    #                     help='number of integration steps (default: 7)')
    parser.add_option('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_option('--bidir', action='store_true', help='enable bidirectional cost function')  # 是否使用双向U-Net架构

    # loss hyperparameters
    parser.add_option('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_option('--lambda', type=float, dest='weight_reg', default=0.01,
                        help='weight of deformation loss (default: 0.01)')


    (options, args) = parser.parse_args()

    return options

# # train list
# train_list = list(range(1,451)) + list(range(651, 688))
# # val list
# val_list = [452]
# # test list
# test_list = list(range(453, 651))
