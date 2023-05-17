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
"""OcclusionSensitivity."""
from __future__ import absolute_import

import numpy as np

from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs

try:
    from tqdm import trange
except (ImportError, AttributeError):
    trange = range
finally:
    pass


class OcclusionSensitivity(Metric):
    """
    Calculates the occlusion sensitivity of the model for a given image, which illustrates which parts of an image are
    most important for a network's classification.

    Occlusion sensitivity refers to how the predicted probability changes with the change of the occluded
    part of an image. The higher the value in the output image is, the greater the decline of certainty, indicating
    that the occluded area is more important in the decision-making process.

    Args:
        pad_val (float): The padding value of the occluded part in an image. Default: ``0.0`` .
        margin (Union[int, Sequence]): Create a cuboid / cube around the voxel you want to occlude. Default: ``2`` .
        n_batch (int): number of images in a batch. Default: ``128`` .
        b_box (Sequence): Bounding box on which to perform the analysis. The output image will also match in size.
                          There should be a minimum and maximum for all dimensions except batch:
                          ``[min1, max1, min2, max2,...]``. If no bounding box is supplied, this will be the same size
                          as the input image. If a bounding box is used, the output image will be cropped to this size.
                          Default: ``None`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>> from mindspore.train import OcclusionSensitivity
        >>>
        >>> class DenseNet(nn.Cell):
        ...     def __init__(self):
        ...         super(DenseNet, self).__init__()
        ...         w = np.array([[0.1, 0.8, 0.1, 0.1],[1, 1, 1, 1]]).astype(np.float32)
        ...         b = np.array([0.3, 0.6]).astype(np.float32)
        ...         self.dense = nn.Dense(4, 2, weight_init=Tensor(w), bias_init=Tensor(b))
        ...
        ...     def construct(self, x):
        ...         return self.dense(x)
        >>>
        >>> model = DenseNet()
        >>> test_data = np.array([[0.1, 0.2, 0.3, 0.4]]).astype(np.float32)
        >>> label = np.array(1).astype(np.int32)
        >>> metric = OcclusionSensitivity()
        >>> metric.clear()
        >>> metric.update(model, test_data, label)
        >>> score = metric.eval()
        >>> print(score)
        [0.29999995    0.6    1.    0.9]
    """
    def __init__(self, pad_val=0.0, margin=2, n_batch=128, b_box=None):
        super().__init__()
        self.pad_val = validator.check_value_type("pad_val", pad_val, [float])
        self.margin = validator.check_value_type("margin", margin, [int, list])
        self.n_batch = validator.check_value_type("n_batch", n_batch, [int])
        self.b_box = b_box if b_box is None else validator.check_value_type("b_box", b_box, [list])
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._baseline = 0
        self._sensitivity_im = 0
        self._is_update = False

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates input, including `model`, `y_pred` and `label`.

        Args:
            inputs: `y_pred` and `label` are a Tensor, list or numpy.ndarray.
                `y_pred`: a batch of images to test, which could be 2D or 3D.
                `label`: classification labels to check for changes. `label` is normally the true label, but
                doesn't have to be.
                `model` is the neural network.

        Raises:
            ValueError: If the number of inputs is not 3.
            RuntimeError: If y_pred.shape[0] is not 1.
            RuntimeError: If the number of labels is different from the number of batches.
        """
        if len(inputs) != 3:
            raise ValueError("For 'OcclusionSensitivity.update', it needs 3 inputs (classification model, "
                             "predicted value, label), but got {}.".format(len(inputs)))

        model = inputs[0]
        y_pred = self._convert_data(inputs[1])
        label = self._convert_data(inputs[2])
        model = validator.check_value_type("model", model, [nn.Cell])

        if y_pred.shape[0] > 1:
            raise RuntimeError(f"For 'OcclusionSensitivity.update', the shape at index 0 of the predicted value "
                               f"(input[1]) must be 1, but got {y_pred.shape[0]}.")

        if isinstance(label, int):
            label = np.array([[label]], dtype=int)
        # If the label is a tensor, make sure  there's only 1 element
        elif np.prod(label.shape) != y_pred.shape[0]:
            raise RuntimeError(f"For 'OcclusionSensitivity.update', the number of the label (input[2]) must be "
                               f"same as the batches, but got the label number {np.prod(label.shape)}, "
                               f"and batches {y_pred.shape[0]}.")

        y_pred_shape = np.array(y_pred.shape[1:])
        b_box_min, b_box_max = _check_input_bounding_box(self.b_box, y_pred_shape)

        temp = model(Tensor(y_pred)).asnumpy()
        self._baseline = temp[0, label].item()

        batch_images = []
        batch_ids = []

        sensitivity_im = np.empty(0, dtype=float)

        output_im_shape = y_pred_shape if self.b_box is None else b_box_max - b_box_min + 1
        num_required_predictions = np.prod(output_im_shape)

        for i in trange(num_required_predictions):
            idx = np.unravel_index(i, output_im_shape)
            if b_box_min is not None:
                idx += b_box_min

            min_idx = [max(0, i - self.margin) for i in idx]
            max_idx = [min(j, i + self.margin) for i, j in zip(idx, y_pred_shape)]

            occlu_im = y_pred.copy()
            occlu_im[(...,) + tuple(slice(i, j) for i, j in zip(min_idx, max_idx))] = self.pad_val

            batch_images.append(occlu_im)
            batch_ids.append(label)

            if len(batch_images) == self.n_batch or i == num_required_predictions - 1:
                sensitivity_im = _append_to_sensitivity_im(model, batch_images, batch_ids, sensitivity_im)
                batch_images = []
                batch_ids = []

        self._sensitivity_im = sensitivity_im.reshape(output_im_shape)
        self._is_update = True

    def eval(self):
        """
         Computes the occlusion_sensitivity.

         Returns:
             A numpy ndarray.

         Raises:
            RuntimeError: If the update method is not called first, an error will be reported.

        """
        if not self._is_update:
            raise RuntimeError("Please call the 'update' method before calling 'eval' method.")

        sensitivity = self._baseline - np.squeeze(self._sensitivity_im)

        return sensitivity


def _append_to_sensitivity_im(model, batch_images, batch_ids, sensitivity_im):
    """
    For a given number of images, the probability of predicting a given label is obtained. Attach to previous
    assessment.
    """
    batch_images = np.vstack(batch_images)
    batch_ids = np.expand_dims(batch_ids, 1)
    model_numpy = model(Tensor(batch_images)).asnumpy()
    first_indices = np.arange(batch_ids.shape[0])[:, None]
    scores = model_numpy[first_indices, batch_ids]
    if sensitivity_im.size == 0:
        return np.vstack(scores)
    return np.vstack((sensitivity_im, scores))


def _check_input_bounding_box(b_box, im_shape):
    """Check that the bounding box (if supplied) is as expected."""
    # If no bounding box has been supplied, set min and max to None
    if b_box is None:
        b_box_min = b_box_max = None
    else:
        if len(b_box) != 2 * len(im_shape):
            raise ValueError(f"For 'OcclusionSensitivity', the bounding box must contain upper and lower for "
                             f"all dimensions (except batch number), and the length of 'b_box' must be twice "
                             f"as long as predicted value's (except batch number), but got 'b_box' length "
                             f"{len(b_box)}, predicted value length (except batch number) {len(im_shape)}.")

        b_box_min = np.array(b_box[::2])
        b_box_max = np.array(b_box[1::2])
        b_box_min[b_box_min < 0] = 0
        b_box_max[b_box_max < 0] = im_shape[b_box_max < 0] - 1
        if np.any(b_box_max >= im_shape):
            raise ValueError("For 'OcclusionSensitivity', maximum bounding box must be smaller than image size "
                             "for all values.")
        if np.any(b_box_min > b_box_max):
            raise ValueError("For 'OcclusionSensitivity', minimum bounding box must be smaller than maximum "
                             "bounding box for all values.")

    return b_box_min, b_box_max
