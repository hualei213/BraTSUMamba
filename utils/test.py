import monai.metrics
import numpy
import numpy as np
from medpy import metric


# calculate two objects distance
def get_ASD(mask_pred, mask_gt):
    # refer : http://loli.github.io/medpy/metric.html
    asd = metric.binary.asd(mask_pred, mask_gt)
    return asd


# calculate mean surface distance
def get_ASSD(mask_pred, mask_gt):
    # refer : http://loli.github.io/medpy/metric.html
    asd = metric.binary.asd(mask_pred, mask_gt, (1.0, 1.0, 1.0))
    return asd


# calculate root mean square surface distance
def get_RMSD(mask_pred, mask_gt):
    # two objects distacne
    pred_gt_distance = metric.binary.__surface_distances(mask_pred, mask_gt)
    pred_gt_square_distance = pred_gt_distance ** 2

    gt_pred_distance = metric.binary.__surface_distances(mask_gt, mask_pred)
    gt_pred_square_distance = gt_pred_distance ** 2

    return np.sqrt(numpy.mean((pred_gt_square_distance.mean(), gt_pred_square_distance.mean())))


def get_percentile_distance(mask_pred, mask_gt, percentile=95):
    pred_gt_distance = metric.binary.__surface_distances(mask_pred, mask_gt)
    gt_pred_distance = metric.binary.__surface_distances(mask_gt, mask_pred)
    result = np.concatenate([pred_gt_distance, gt_pred_distance])
    return np.percentile(result, percentile)

def get_percentile_distance_another(mask_pred, mask_gt, percentile=95):
    mask_pred = np.expand_dims(mask_pred,axis=[0,1])
    mask_gt = np.expand_dims(mask_gt,axis=[0,1])
    # print(mask_pred.shape)
    hd = monai.metrics.compute_hausdorff_distance(y_pred=mask_pred,y=mask_gt,percentile=percentile,directed=False)
    return hd.item()

    # result = np.concatenate([pred_gt_distance, gt_pred_distance])
    # return np.percentile(result, percentile)


def get_hsdf(mask_pred, mask_gt):
    pred_gt_distance = metric.binary.__surface_distances(mask_pred, mask_gt)
    gt_pred_distance = metric.binary.__surface_distances(mask_gt, mask_pred)
    result = np.concatenate([pred_gt_distance, gt_pred_distance])
    return result.max()

def get_hd(mask_pred, mask_gt):
    mask_pred = np.expand_dims(mask_pred, axis=[0, 1])
    mask_gt = np.expand_dims(mask_gt, axis=[0, 1])
    hd = monai.metrics.compute_hausdorff_distance(y_pred=mask_pred, y=mask_gt)
    return hd.item()

def get_sens(mask_pred, mask_gt):
    return metric.sensitivity(mask_pred,mask_gt)

def get_spec(mask_pred, mask_gt):
    return metric.specificity(mask_pred,mask_gt)

