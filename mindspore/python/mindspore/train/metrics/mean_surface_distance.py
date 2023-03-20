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
from __future__ import absolute_import

from scipy.ndimage import morphology
import numpy as np

from mindspore._checkparam import Validator as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs


class MeanSurfaceDistance(Metric):
    r"""
    Computes the Average Surface Distance from `y_pred` to `y` under the default setting. It measures how much,
    on average, the surface varies between the segmentation and the GT (ground truth).

    Given two sets A and B, S(A) denotes the set of surface voxels of A, the shortest distance of an arbitrary voxel v
    to S(A) is defined as:

    .. math::
        {\text{dis}}\left (v, S(A)\right ) = \underset{s_{A}  \in S(A)}{\text{min }}\rVert v - s_{A} \rVert

    The Average Surface Distance from set(B) to set(A) is given by:

    .. math::
        AvgSurDis(B \rightarrow A) = \frac{\sum_{s_{B}  \in S(B)}^{} {\text{dis}  \left
        ( s_{B}, S(A) \right )} } {\left | S(B) \right |}

    Where the \|\|\*\|\| denotes a distance measure. \|\*\| denotes the number of elements.

    The mean of surface distance from set(B) to set(A) and from set(A) to set(B) is:

    .. math::
        MeanSurDis(A \leftrightarrow B) = \frac{\sum_{s_{A}  \in S(A)}^{} {\text{dis}  \left ( s_{A}, S(B) \right )}
        + \sum_{s_{B}  \in S(B)}^{} {\text{dis}  \left ( s_{B}, S(A) \right )} }{\left | S(A) \right | +
        \left | S(B) \right |}

    Args:
        distance_metric (string): Three measurement methods are supported: "euclidean", "chessboard" or "taxicab".
                          Default: "euclidean".
        symmetric (bool): Whether to calculate the Mean Surface Distance between y_pred and y.
                          If False, it only calculates :math:`AvgSurDis({y\_pred} \rightarrow y)`,
                          otherwise, the mean of  distance from `y_pred` to `y` and from `y` to `y_pred`, i.e.
                          :math:`MeanSurDis(y\_pred \leftrightarrow y)`, will be returned. Default: False.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import MeanSurfaceDistance
        >>> x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
        >>> y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
        >>> metric = MeanSurfaceDistance(symmetric=False, distance_metric="euclidean")
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
        self.distance_metric = validator.check_string(distance_metric, self.distance_metric_list, "distance_metric")
        self.symmetric = validator.check_value_type("symmetric", symmetric, [bool])
        self.clear()
        self._is_update = None
        self._y_edges = None
        self._y_pred_edges = None

    def clear(self):
        """Clears the internal evaluation result."""
        self._y_pred_edges = 0
        self._y_edges = 0
        self._is_update = False

    def _get_surface_distance(self, y_pred_edges, y_edges):
        """
        Calculate the surface distances from `y_pred_edges` to `y_edges`.

         Args:
            y_pred_edges (np.ndarray): the edge of the predictions.
            y_edges (np.ndarray): the edge of the ground truth.
        """
        if not np.any(y_pred_edges):
            return np.array([])

        if np.any(y_edges):
            if self.distance_metric == "euclidean":
                dis = morphology.distance_transform_edt(~y_edges)
            elif self.distance_metric in self.distance_metric_list[-2:]:
                dis = morphology.distance_transform_cdt(~y_edges, metric=self.distance_metric)
        else:
            dis = np.full(y_edges.shape, np.inf)

        return dis[y_pred_edges]

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result 'y_pred', 'y' and 'label_idx'.

        Args:
            inputs: Input 'y_pred', 'y' and 'label_idx'. 'y_pred' and 'y' are a Tensor, list or numpy.ndarray.
                    'y_pred' is the predicted binary image. 'y' is the actual binary image. 'label_idx', the data
                    type of `label_idx` is int.

        Raises:
            ValueError: If the number of the inputs is not 3.
            TypeError: If the data type of label_idx is not int or float.
            ValueError: If the value of label_idx is not in y_pred or y.
            ValueError: If y_pred and y have different shapes.
        """
        if len(inputs) != 3:
            raise ValueError("For 'MeanSurfaceDistance.update', it needs 3 inputs (predicted value, true value, "
                             "label index), but got {}".format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        label_idx = inputs[2]

        if not isinstance(label_idx, (int, float)):
            raise ValueError(f"For 'MeanSurfaceDistance.update', the label index (input[2]) must be int or float, "
                             f"but got {type(label_idx)}.")

        if label_idx not in y_pred and label_idx not in y:
            raise ValueError("For 'MeanSurfaceDistance.update', the label index (input[2]) must be in predicted "
                             "value (input[0]) or true value (input[1]), but {} is not.".format(label_idx))

        if y_pred.size == 0 or y_pred.shape != y.shape:
            raise ValueError(f"For 'MeanSurfaceDistance.update', the size of predicted value (input[0]) and true "
                             f"value (input[1]) must be greater than 0, in addition to that, predicted value and "
                             f"true value must have the same shape, but got predicted value size: {y_pred.size}, "
                             f"shape: {y_pred.shape}, true value size: {y.size}, shape: {y.shape}.")

        if y_pred.dtype != bool:
            y_pred = y_pred == label_idx
        if y.dtype != bool:
            y = y == label_idx

        self._y_pred_edges = morphology.binary_erosion(y_pred) ^ y_pred
        self._y_edges = morphology.binary_erosion(y) ^ y
        self._is_update = True

    def eval(self):
        """
        Calculate mean surface distance.

        Returns:
             numpy.float64. The mean surface distance value.

        Raises:
            RuntimeError: If the update method is not called first, an error will be reported.
        """
        if self._is_update is False:
            raise RuntimeError("Please call the 'update' method before calling 'eval' method.")

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
