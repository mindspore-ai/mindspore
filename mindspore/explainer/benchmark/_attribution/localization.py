# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Localization metrics."""
import numpy as np

from mindspore.train._utils import check_value_type
from .metric import LabelSensitiveMetric
from ..._operators import maximum, reshape, Tensor
from ..._utils import format_tensor_to_ndarray


def _get_max_position(saliency):
    """Get the position of the max pixel of the saliency map."""
    saliency = saliency.asnumpy()
    w = saliency.shape[3]
    saliency = np.reshape(saliency, (len(saliency), -1))
    max_arg = np.argmax(saliency, axis=1)
    return max_arg // w, max_arg - (max_arg // w) * w


def _mask_out_saliency(saliency, threshold):
    """Keep the saliency map with value greater than threshold."""
    max_value = maximum(saliency)
    mask_out = saliency > (reshape(max_value, (len(saliency), -1, 1, 1)) * threshold)
    return mask_out


class Localization(LabelSensitiveMetric):
    r"""
    Provides evaluation on the localization capability of XAI methods.

    Three specific metrics to obtain quantified results are supported: "PointingGame", and "IoSR"
    (Intersection over Salient Region).

    For metric "PointingGame", the localization capability is calculated as the ratio of data in which the max position
    of their saliency maps lies within the bounding boxes. Specifically, for a single datum, given the saliency map and
    its bounding box, if the max point of its saliency map lies within the bounding box, the evaluation result is 1
    otherwise 0.

    For metric "IoSR" (Intersection over Salient Region), the localization capability is calculated as the intersection
    of the bounding box and the salient region over the area of the salient region. The salient region is defined as
    the region whose value exceeds :math:`\theta * \max{saliency}`.

    Args:
        num_labels (int): Number of classes in the dataset.
        metric (str, optional): Specific metric to calculate localization capability.
            Options: "PointingGame", "IoSR". Default: "PointingGame".
    """

    def __init__(self,
                 num_labels,
                 metric="PointingGame"
                 ):
        super(Localization, self).__init__(num_labels)
        self._verify_metrics(metric)
        self._metric = metric

        # Arg for specific metric, for "PointingGame" it should be an integer indicating the tolerance
        # of "PointingGame", while for "IoSR" it should be a float number
        # indicating the threshold to choose salient region. Default: 25.
        if self._metric == "PointingGame":
            self._metric_arg = 15
        else:
            self._metric_arg = 0.5

    @staticmethod
    def _verify_metrics(metric):
        """Verify the user defined metric."""
        supports = ["PointingGame", "IoSR"]
        if metric not in supports:
            raise ValueError("Metric should be one of {}".format(supports))

    def evaluate(self, explainer, inputs, targets, saliency=None, mask=None):
        """
        Evaluate localization on a single data sample.

        Note:
             Currently only single sample (:math:`N=1`) at each call is supported.

        Args:
            explainer (Explanation): The explainer to be evaluated, see `mindspore.explainer.explanation`.
            inputs (Tensor): A data sample, a 4D tensor of shape :math:`(N, C, H, W)`.
            targets (Tensor, int): The label of interest. It should be a 1D or 0D tensor, or an integer.
                If `targets` is a 1D tensor, its length should be the same as `inputs`.
            saliency (Tensor, optional): The saliency map to be evaluated, a 4D tensor of shape :math:`(N, 1, H, W)`.
                If it is None, the parsed `explainer` will generate the saliency map with `inputs` and `targets` and
                continue the evaluation. Default: None.
            mask (Tensor, numpy.ndarray): Ground truth bounding box/masks for the inputs w.r.t targets, a 4D tensor
                or numpy.ndarray of shape :math:`(N, 1, H, W)`.

        Returns:
            numpy.ndarray, 1D array of shape :math:`(N,)`, result of localization evaluated on `explainer`.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore.explainer.explanation import Gradient
            >>> from mindspore.explainer.benchmark import Localization
            >>>
            >>> num_labels = 10
            >>> localization = Localization(num_labels, "PointingGame")
            >>>
            >>> # The detail of LeNet5 is shown in model_zoo.official.cv.lenet.src.lenet.py
            >>> net = LeNet5(10, num_channel=3)
            >>> gradient = Gradient(net)
            >>> inputs = ms.Tensor(np.random.rand(1, 3, 32, 32), ms.float32)
            >>> masks = np.zeros([1, 1, 32, 32])
            >>> masks[:, :, 10: 20, 10: 20] = 1
            >>> targets = 5
            >>> # usage 1: input the explainer and the data to be explained,
            >>> # localization is a Localization instance
            >>> res = localization.evaluate(gradient, inputs, targets, mask=masks)
            >>> print(res.shape)
            (1,)
            >>> # usage 2: input the generated saliency map
            >>> saliency = gradient(inputs, targets)
            >>> res = localization.evaluate(gradient, inputs, targets, saliency, mask=masks)
            >>> print(res.shape)
            (1,)
        """
        self._check_evaluate_param_with_mask(explainer, inputs, targets, saliency, mask)

        mask_np = format_tensor_to_ndarray(mask)[0]

        if saliency is None:
            saliency = explainer(inputs, targets)

        if self._metric == "PointingGame":
            point = _get_max_position(saliency)

            x, y = np.meshgrid(
                (np.arange(mask_np.shape[1]) - point[0]) ** 2,
                (np.arange(mask_np.shape[2]) - point[1]) ** 2)
            max_region = (x + y) < self._metric_arg ** 2

            # if max_region has overlap with mask_np return 1 otherwise 0.
            result = 1 if (mask_np.astype(bool) & max_region).any() else 0

        elif self._metric == "IoSR":
            mask_out = _mask_out_saliency(saliency, self._metric_arg)
            mask_out_np = format_tensor_to_ndarray(mask_out)
            overlap = np.sum(mask_np.astype(bool) & mask_out_np.astype(bool))
            saliency_area = np.sum(mask_out_np)
            result = overlap / saliency_area.clip(min=1e-10)
        return np.array([result], np.float)

    def _check_evaluate_param_with_mask(self, explainer, inputs, targets, saliency, mask):
        self._check_evaluate_param(explainer, inputs, targets, saliency)
        if len(inputs.shape) != 4:
            raise ValueError('Argument mask must be 4D Tensor')
        if mask is None:
            raise ValueError('To compute localization, mask must be provided.')
        check_value_type('mask', mask, (Tensor, np.ndarray))
        if len(mask.shape) != 4 or len(mask) != len(inputs):
            raise ValueError("The input mask must be 4-dimensional (1, 1, h, w) with same length of inputs.")
