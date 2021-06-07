# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MeanSurfaceDistance."""
import numpy as np
from mindspore._checkparam import Validator as validator
from .metric import Metric


class MeanSurfaceDistance(Metric):
    """
    This function is used to compute the Average Surface Distance from `y_pred` to `y` under the default setting.
    Mean Surface Distance(MSD), the mean of the vector is taken. This tell us how much, on average, the surface varies
    between the segmentation and the GT.

    Args:
        distance_metric (string): The parameter of calculating Hausdorff distance supports three measurement methods,
                                  "euclidean", "chessboard" or "taxicab". Default: "euclidean".
        symmetric (bool): if calculate the symmetric average surface distance between `y_pred` and `y`. In addition,
                          if sets ``symmetric = True``, the average symmetric surface distance between these two inputs
                          will be returned. Defaults: False.

    Examples:
        >>> x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
        >>> y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
        >>> metric = nn.MeanSurfaceDistance(symmetric=False, distance_metric="euclidean")
        >>> metric.clear()
        >>> metric.update(x, y, 0)
        >>> mean_average_distance = metric.eval()
        >>> print(mean_average_distance)
        0.8047378541243649
    """

    def __init__(self, symmetric=False, distance_metric="euclidean"):
        super(MeanSurfaceDistance, self).__init__()
        self.distance_metric_list = ["euclidean", "chessboard", "taxicab"]
        distance_metric = validator.check_value_type("distance_metric", distance_metric, [str])
        validator.check_string(distance_metric, self.distance_metric_list, "distance_metric")
        self.distance_metric = distance_metric
        self.symmetric = validator.check_value_type("symmetric", symmetric, [bool])
        self._y_pred_edges = 0
        self._y_edges = 0
        self._is_update = False
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._y_pred_edges = 0
        self._y_edges = 0
        self._is_update = False

    def update(self, *inputs):
        """
        Updates the internal evaluation result 'y_pred', 'y' and 'label_idx'.

        Args:
            inputs: Input 'y_pred', 'y' and 'label_idx'. 'y_pred' and 'y' are Tensor or numpy.ndarray. 'y_pred' is the
                    predicted binary image. 'y' is the actual binary image. 'label_idx', the data type of `label_idx`
                    is int.

         Raises:
            ValueError: If the number of the inputs is not 3.
        """
        if len(inputs) != 3:
            raise ValueError('MeanSurfaceDistance need 3 inputs (y_pred, y, label), but got {}.'.format(len(inputs)))

        self._y_pred_edges, self._y_edges = self._check_surface_distance_inputs(inputs)
        self._is_update = True

    def eval(self):
        """
        Calculate mean surface distance.
        """
        if self._is_update is False:
            raise RuntimeError('Call the update method before calling eval.')

        mean_surface_distance = Metric._get_surface_distance(self._y_pred_edges, self._y_edges, self.distance_metric)

        if mean_surface_distance.shape == (0,):
            return np.inf

        avg_surface_distance = mean_surface_distance.mean()

        if not self.symmetric:
            return avg_surface_distance

        contrary_mean_surface_distance = Metric._get_surface_distance(self._y_edges, self._y_pred_edges,
                                                                      self.distance_metric)
        if contrary_mean_surface_distance.shape == (0,):
            return np.inf

        contrary_avg_surface_distance = contrary_mean_surface_distance.mean()
        return np.mean((avg_surface_distance, contrary_avg_surface_distance))
