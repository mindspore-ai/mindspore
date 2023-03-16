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
"""HausdorffDistance."""
from __future__ import absolute_import

from collections import abc
from abc import ABCMeta
from scipy.ndimage import morphology
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs


class _ROISpatialData(metaclass=ABCMeta):
    """
    Produce Region Of Interest (ROI). Support to crop ND spatial data. The center and size of the space should be
    provided, if not, the start and end coordinates of the ROI must be provided.

    Args:
        roi_center (int): The central coordinates of the crop ROI.
        roi_size (int): The size of the crop ROI.
        roi_start (int): The start coordinates of the crop ROI.
        roi_end (int): The end coordinates of the crop ROI.
    """

    def __init__(self, roi_center=None, roi_size=None, roi_start=None, roi_end=None):

        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.int16)
            roi_size = np.asarray(roi_size, dtype=np.int16)
            self.roi_start = np.maximum(roi_center - np.floor_divide(roi_size, 2), 0)
            self.roi_end = np.maximum(self.roi_start + roi_size, self.roi_start)
        else:
            if roi_start is None or roi_end is None:
                raise ValueError("For 'HausdorffDistance.update', When either 'roi_center' or 'roi_size' is None,"
                                 "neither 'roi_start' nor 'roi_end' can be None.")
            self.roi_start = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
            self.roi_end = np.maximum(np.asarray(roi_end, dtype=np.int16), self.roi_start)

    def __call__(self, data):
        """
        Transform the data, if the data is channel first, slicing is not applicable to channel dim.

        Args:
            data (np.ndarray): Data to be converted.

        Returns:
            np.ndarray, transform result.
        """
        sd = min(len(self.roi_start), len(self.roi_end), len(data.shape[1:]))
        slices = [slice(None)] + [slice(s, e) for s, e in zip(self.roi_start[:sd], self.roi_end[:sd])]
        return data[tuple(slices)]


class HausdorffDistance(Metric):
    r"""
    Calculates the Hausdorff distance. Hausdorff distance is the maximum and minimum distance between two point sets.
    Given two feature sets A and B, the Hausdorff distance between two point sets A and B is defined as follows:

    .. math::
        \begin{array}{ll} \\
                H(A, B) = \text{max}[h(A, B), h(B, A)]\\
                h(A, B) = \underset{a \in A}{\text{max}}\{\underset{b \in B}{\text{min}} \rVert a - b \rVert \}\\
                h(B, A) = \underset{b \in B}{\text{max}}\{\underset{a \in A}{\text{min}} \rVert b - a \rVert \}
        \end{array}

    where :math:`h(A,B)` is the maximum distance of a set A to the nearest point in the set B,
    :math:`h(B, A)` is the maximum distance
    of a set B to the nearest point in the set A. The distance calculation is oriented, which means that most of times
    :math:`h(A, B)` is not equal to :math:`h(B, A)`. :math:`H(A, B)` is the two-way Hausdorff distance.

    Args:
        distance_metric (string): Three distance measurement methods are supported: "euclidean", "chessboard" or
                           "taxicab". Default: "euclidean".
        percentile (float): Floating point numbers between 0 and 100. Specify the percentile parameter to get the
                            percentile of the Hausdorff distance. Default: None.
        directed (bool): If True, it only calculates h(y_pred, y) distance, otherwise, max(h(y_pred, y), h(y, y_pred))
                    will be returned. Default: False.
        crop (bool): Crop input images and only keep the foregrounds. In order to maintain two inputs' shapes,
                     here the bounding box is achieved by (y_pred | y) which represents the union set of two images.
                     Default: True.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import HausdorffDistance
        >>>
        >>> x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
        >>> y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
        >>> metric = HausdorffDistance()
        >>> metric.clear()
        >>> metric.update(x, y, 0)
        >>> mean_average_distance = metric.eval()
        >>> print(mean_average_distance)
        1.4142135623730951
    """
    def __init__(self, distance_metric="euclidean", percentile=None, directed=False, crop=True):
        super(HausdorffDistance, self).__init__()
        string_list = ["euclidean", "chessboard", "taxicab"]
        distance_metric = validator.check_value_type("distance_metric", distance_metric, [str])
        self.distance_metric = validator.check_string(distance_metric, string_list, "distance_metric")
        if percentile is None:
            self.percentile = percentile
        else:
            self.percentile = validator.check_value_type("percentile", percentile, [float])
        self.directed = directed if directed is None else validator.check_value_type("directed", directed, [bool])
        self.crop = crop if crop is None else validator.check_value_type("crop", crop, [bool])
        self.clear()

    def _is_tuple_rep(self, tup, dim):
        """
        Returns the tup containing the dim value by shortening or repeating the input.

        Raises:
            ValueError: When tup is a sequence and tup length is not dim.

        """
        result = None
        if not _is_iterable_sequence(tup):
            result = (tup,) * dim
        elif len(tup) == dim:
            result = tuple(tup)

        if result is None:
            raise ValueError(f"The sequence length must be {dim}, but got {len(tup)}.")

        return result

    def _is_tuple(self, inputs):
        """
        Returns a tuple of inputs.
        """
        if not _is_iterable_sequence(inputs):
            inputs = (inputs,)

        return tuple(inputs)

    def _create_space_bounding_box(self, image, func=lambda x: x > 0, channel_indices=None, margin=0):
        """
        The position of the space bounding box that generates the foreground in an image with start end.
        The user can define any function to select the desired foreground from the whole image or the specified channel.
        It can also add margins to each size of the bounding box.

        Args:
            image: source image to generate bounding box from.
            func: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only a single value is provided,
                    use it for all dims.
        """
        data = image[[*(self._is_tuple(channel_indices))]] if channel_indices is not None else image
        data = np.any(func(data), axis=0)
        nonzero_idx = np.nonzero(data)
        margin = self._is_tuple_rep(margin, data.ndim)

        box_start = list()
        box_end = list()
        for i in range(data.ndim):
            if nonzero_idx[i].size <= 0:
                raise ValueError("Did not find nonzero index at the spatial dim {}".format(i))
            box_start.append(max(0, np.min(nonzero_idx[i]) - margin[i]))
            box_end.append(min(data.shape[i], np.max(nonzero_idx[i]) + margin[i] + 1))
        return box_start, box_end

    def _calculate_percent_hausdorff_distance(self, y_pred_edges, y_edges):
        """
        Calculate the directed Hausdorff distance.

        Args:
            y_pred_edges (np.ndarray): the edge of the predictions.
            y_edges (np.ndarray): the edge of the ground truth.
        """
        surface_distance = self._get_surface_distance(y_pred_edges, y_edges)

        if surface_distance.shape == (0,):
            return np.inf

        if not self.percentile:
            return surface_distance.max()
        if 0 <= self.percentile <= 100:
            return np.percentile(surface_distance, self.percentile)

        raise ValueError(f"For 'HausdorffDistance', the value of the argument 'percentile' must be [0, 100], "
                         f"but got {self.percentile}.")

    def _get_surface_distance(self, y_pred_edges, y_edges):
        """
        Calculate the surface distances from `y_pred_edges` to `y_edges`.

         Args:
            y_pred_edges (np.ndarray): the edge of the predictions.
            y_edges (np.ndarray): the edge of the ground truth.
        """

        if not np.any(y_pred_edges):
            return np.array([])

        if not np.any(y_edges):
            dis = np.inf * np.ones_like(y_edges)
        else:
            if self.distance_metric == "euclidean":
                dis = morphology.distance_transform_edt(~y_edges)
            elif self.distance_metric == "chessboard" or self.distance_metric == "taxicab":
                dis = morphology.distance_transform_cdt(~y_edges, metric=self.distance_metric)

        surface_distance = dis[y_pred_edges]

        return surface_distance

    def _get_mask_edges_distance(self, y_pred, y):
        """
        Do binary erosion and use XOR for input to get the edges. This function is helpful to further
        calculate metrics such as Average Surface Distance and Hausdorff Distance.

         Args:
            y_pred (np.ndarray): the edge of the predictions.
            y (np.ndarray): the edge of the ground truth.
        """
        if self.crop:
            if not np.any(y_pred | y):
                res1 = np.zeros_like(y_pred)
                res2 = np.zeros_like(y)
                return res1, res2

            y_pred, y = np.expand_dims(y_pred, 0), np.expand_dims(y, 0)
            box_start, box_end = self._create_space_bounding_box(y_pred | y)
            cropper = _ROISpatialData(roi_start=box_start, roi_end=box_end)
            y_pred, y = np.squeeze(cropper(y_pred)), np.squeeze(cropper(y))

        y_pred = morphology.binary_erosion(y_pred) ^ y_pred
        y = morphology.binary_erosion(y) ^ y

        return y_pred, y

    def clear(self):
        """Clears the internal evaluation result."""
        self.y_pred_edges = 0
        self.y_edges = 0
        self._is_update = False

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result with the inputs: 'y_pred', 'y' and 'label_idx'.

        Args:
            inputs: Input 'y_pred', 'y' and 'label_idx'. 'y_pred' and 'y' are a `Tensor`, list or
                        numpy.ndarray.  'y_pred' is the predicted binary image. 'y' is the actual
                        binary image. Data type of 'label_idx' is int or float.

        Raises:
            ValueError: If the number of the inputs is not 3.
            TypeError: If the data type of label_idx is not int or float.
            ValueError: If the value of label_idx is not in y_pred or y.
            ValueError: If y_pred and y have different shapes.
        """
        self._is_update = True

        if len(inputs) != 3:
            raise ValueError("For 'HausdorffDistance.update', it needs 3 inputs (predicted value, true value, "
                             "label index), but got {}.".format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        label_idx = inputs[2]

        if not isinstance(label_idx, (int, float)):
            raise ValueError(f"For 'HausdorffDistance.update', the label index (input[2]) must be int or float, "
                             f"but got {type(label_idx)}.")

        if label_idx not in y_pred and label_idx not in y:
            raise ValueError("For 'HausdorffDistance.update', the label index (input[2]) must be in predicted "
                             "value (input[0]) or true value (input[1]), but {} is not.".format(label_idx))

        if y_pred.size == 0 or y_pred.shape != y.shape:
            raise ValueError(f"For 'HausdorffDistance.update', the size of predicted value (input[0]) and true value "
                             f"(input[1]) must be greater than 0, in addition to that, predicted value and true "
                             f"value should have the same shape, but got predicted value size: {y_pred.size}, shape: "
                             f"{y_pred.shape}, true value size: {y.size}, shape: {y.shape}.")

        y_pred = (y_pred == label_idx) if y_pred.dtype is not bool else y_pred
        y = (y == label_idx) if y.dtype is not bool else y

        self.y_pred_edges, self.y_edges = self._get_mask_edges_distance(y_pred, y)

    def eval(self):
        """
        Calculate the no-directed or directed Hausdorff distance.

        Returns:
             numpy.float64, the hausdorff distance.

        Raises:
            RuntimeError: If the update method is not called first, an error will be reported.
        """
        if self._is_update is False:
            raise RuntimeError("Please call the 'update' method before calling 'eval' method.")

        hd = self._calculate_percent_hausdorff_distance(self.y_pred_edges, self.y_edges)
        if self.directed:
            return hd

        hd2 = self._calculate_percent_hausdorff_distance(self.y_edges, self.y_pred_edges)
        return max(hd, hd2)


def _is_iterable_sequence(inputs):
    """
    Determine if the input is an iterable sequence and it is not a string.
    """
    if isinstance(inputs, Tensor):
        return int(inputs.dim()) > 0
    return isinstance(inputs, abc.Iterable) and not isinstance(inputs, str)
