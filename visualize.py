import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import get_args, test_list

"""Visualize results by slices.
"""

################################################################################
# Arguments
################################################################################
args = get_args()

LABEL_DIR = args.test_label_dir
TEST_DIR = args.result_dir
PATCH_SIZE = args.patch_size
CHECKPOINT_NUM = args.checkpoint_num
OVERLAP_STEPSIZE = args.overlap_step
TEST_ID = test_list


################################################################################
# Functions
################################################################################
def Visualize(label_dir, test_dir, test_id, patch_size, checkpoint_num,
              overlap_step):
    fig_folder = os.path.join(test_dir, 'visualize',
                              str(test_id),
                              str(checkpoint_num))
    # fig_folder = os.path.join(test_dir, 'visualize',
    # 						  str(test_id))

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    else:
        print('Figure folder: ' + fig_folder + ' exists. Return.')
        return

    print('Perform visualization for instance-%d:' % test_id)

    print('Loading label...')
    label_file = os.path.join(label_dir, 'instance-%d-label.npy' % test_id)
    assert os.path.isfile(label_file), \
        ('Please generate the label file.')
    label = np.load(label_file)
    print('Check label: ', label.shape, np.max(label))

    print('Loading test result...')
    test_file = os.path.join(test_dir, 'test_%d_checkpoint_%d.npy'
                             % (test_id, checkpoint_num))
    assert os.path.isfile(test_file), \
        ('Run main.py --option=predict to generate the prediction results.')
    test = np.load(test_file)
    print('Check test: ', test.shape, np.max(test))

    # set depth_list
    depth_list = list(range(0, test.shape[2]))

    for slice_depth in depth_list:
        print()
        print('Visualizing slice %d' % slice_depth)
        test_show = test[:, :, slice_depth]
        # test_show = label[:, :, slice_depth]
        label_show = label[:, :, slice_depth]

        fig = plt.figure(figsize=(20, 20))
        fig.suptitle('Compare the %d-th slice.' % slice_depth, fontsize=24)

        a = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(label_show, cmap='gray')
        a.set_title('Groud Truth')

        a = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(test_show, cmap='gray')
        a.set_title('Prediction')

        # fig_path = os.path.join(fig_folder,
        # 						'visualization-sub-%d-slice-%d-overlap-%d-chkpt-%d' % \
        # 						(test_id, slice_depth, overlap_step, checkpoint_num))

        fig_path = os.path.join(fig_folder,
                                'visualization-sub-%d-slice-%d-overlap-%d' % \
                                (test_id, slice_depth, overlap_step))

        plt.savefig(fig_path)


if __name__ == '__main__':

    for test_id_current in TEST_ID:
        Visualize(
            label_dir=LABEL_DIR,
            test_dir=TEST_DIR,
            test_id=test_id_current,
            patch_size=PATCH_SIZE,
            checkpoint_num=CHECKPOINT_NUM,
            overlap_step=OVERLAP_STEPSIZE)
