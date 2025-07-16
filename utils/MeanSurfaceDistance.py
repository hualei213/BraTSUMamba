"""MeanSurfaceDistance."""
# from keras.metrics import Metric
from scipy.ndimage import morphology
import numpy as np
from mindspore._checkparam import Validator as validator
from metric import Metric


class MeanSurfaceDistance(Metric):

    def __init__(self, symmetric=False, distance_metric="euclidean"):
        super(MeanSurfaceDistance, self).__init__()
        self.distance_metric_list = ["euclidean", "chessboard", "taxicab"]
        distance_metric = validator.check_value_type("distance_metric", distance_metric, [str])
        self.distance_metric = validator.check_string(distance_metric, self.distance_metric_list, "distance_metric")
        self.symmetric = validator.check_value_type("symmetric", symmetric, [bool])
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._y_pred_edges = 0
        self._y_edges = 0
        self._is_update = False

    def _get_surface_distance(self, y_pred_edges, y_edges):
        """
        计算从预测图片边界到真实图片边界的表面距离。
        """

        if not np.any(y_pred_edges):
            return np.array([])

        if not np.any(y_edges):
            dis = np.full(y_edges.shape, np.inf)
        else:
            if self.distance_metric == "euclidean":
                dis = morphology.distance_transform_edt(~y_edges)
            elif self.distance_metric in self.distance_metric_list[-2:]:
                dis = morphology.distance_transform_cdt(~y_edges, metric=self.distance_metric)

        surface_distance = dis[y_pred_edges]

        return surface_distance

    def update(self, *inputs):
        """
        更新输入数据。
        """
        if len(inputs) != 3:
            raise ValueError('MeanSurfaceDistance need 3 inputs (y_pred, y, label), but got {}.'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        label_idx = inputs[2]

        if y_pred.size == 0 or y_pred.shape != y.shape:
            raise ValueError("y_pred and y should have same shape, but got {}, {}.".format(y_pred.shape, y.shape))

        if y_pred.dtype != bool:
            y_pred = y_pred == label_idx
        if y.dtype != bool:
            y = y == label_idx

        self._y_pred_edges = morphology.binary_erosion(y_pred) ^ y_pred
        self._y_edges = morphology.binary_erosion(y) ^ y
        self._is_update = True

    def eval(self):
        """
        计算平均表面距离。
        """
        if self._is_update is False:
            raise RuntimeError('Call the update method before calling eval.')

        mean_surface_distance = self._get_surface_distance(self._y_pred_edges, self._y_edges)

        if mean_surface_distance.shape == (0,):
            return np.inf

        avg_surface_distance = mean_surface_distance.mean()

        if not self.symmetric:
            return avg_surface_distance

        contrary_mean_surface_distance = self._get_surface_distance(self._y_edges, self._y_pred_edges)
        if contrary_mean_surface_distance.shape == (0,):
            return np.inf

        contrary_avg_surface_distance = contrary_mean_surface_distance.mean()
        return np.mean((avg_surface_distance, contrary_avg_surface_distance))

def test_mean_surface_distance(pred,target):
    """test_mean_surface_distance"""
    metric = MeanSurfaceDistance()
    metric.clear()
    metric.update(pred, target, 0)
    distance = metric.eval()
    print(distance)
    return distance


