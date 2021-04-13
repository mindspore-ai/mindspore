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
import numpy as np
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from .metric import Metric

try:
    from tqdm import trange
except (ImportError, AttributeError):
    trange = range


class OcclusionSensitivity(Metric):
    """
    This function is used to calculate the occlusion sensitivity of the model for a given image.
    Occlusion sensitivity refers to how the probability of a given prediction changes with the change of the occluded
    part of the image.

    For a given result, the output probability is the probability of a region.

    The higher the value in the output image, the greater the decline of certainty, indicating that
    the occluded area is more important in the decision-making process.

    Args:
        pad_val (float): What values need to be entered in the image when a part of the image is occluded. Default: 0.0.
        margin (Union[int, Sequence]): Create a cuboid / cube around the voxel you want to occlude. Default: 2.
        n_batch (int): number of images in a batch before inference. Default: 128.
        b_box (Sequence): Bounding box on which to perform the analysis. The output image will also match in size.
                          There should be a minimum and maximum for all dimensions except batch:
                          ``[min1, max1, min2, max2,...]``. If no bounding box is supplied, this will be the same size
                          as the input image. If a bounding box is used, the output image will be cropped to this size.
                          Default: None.

    Example:
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
        [0.29999995    0.6    1    0.9]
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

    def _check_input_bounding_box(self, b_box, im_shape):
        """Check that the bounding box (if supplied) is as expected."""
        # If no bounding box has been supplied, set min and max to None
        if b_box is None:
            b_box_min = b_box_max = None
        else:
            if len(b_box) != 2 * len(im_shape):
                raise ValueError("Bounding box should contain upper and lower for all dimensions (except batch number)")

            b_box_min = np.array(b_box[::2])
            b_box_max = np.array(b_box[1::2])
            b_box_min[b_box_min < 0] = 0
            b_box_max[b_box_max < 0] = im_shape[b_box_max < 0] - 1
            if np.any(b_box_max >= im_shape):
                raise ValueError("Max bounding box should be < image size for all values")
            if np.any(b_box_min > b_box_max):
                raise ValueError("Min bounding box should be <= max for all values")

        return b_box_min, b_box_max

    def _append_to_sensitivity_im(self, model, batch_images, batch_ids, sensitivity_im):
        """For a given number of images, the probability of predicting a given label is obtained. Attach to previous
        assessment."""
        batch_images = np.vstack(batch_images)
        batch_ids = np.expand_dims(batch_ids, 1)
        model_numpy = model(Tensor(batch_images)).asnumpy()
        first_indices = np.arange(batch_ids.shape[0])[:, None]
        scores = model_numpy[first_indices, batch_ids]
        if sensitivity_im.size == 0:
            return np.vstack(scores)
        return np.vstack((sensitivity_im, scores))

    def update(self, *inputs):
        """
        Updates input, including `model`, `y_pred` and `label`.

        Inputs:
            - **model** (nn.Cell) - classification model to use for inference.
            - **y_pred** (Union[Tensor, list, np.ndarray]) - image to test. Should be tensor consisting of 1 batch,
              can be 2- or 3D.
            - **label** (Union[int, Tensor]) - classification label to check for changes (normally the true label,
              but doesn't have to be

        Raises:
            ValueError: If the number of input is not 3.
        """
        if len(inputs) != 3:
            raise ValueError('occlusion_sensitivity need 3 inputs (model, y_pred, y), but got {}'.format(len(inputs)))

        model = inputs[0]
        y_pred = self._convert_data(inputs[1])
        label = self._convert_data(inputs[2])
        model = validator.check_value_type("model", model, [nn.Cell])

        if y_pred.shape[0] > 1:
            raise RuntimeError("Expected batch size of 1.")

        if isinstance(label, int):
            label = np.array([[label]], dtype=int)
        # If the label is a tensor, make sure  there's only 1 element
        elif np.prod(label.shape) != y_pred.shape[0]:
            raise RuntimeError("Expected as many labels as batches.")

        y_pred_shape = np.array(y_pred.shape[1:])
        b_box_min, b_box_max = self._check_input_bounding_box(self.b_box, y_pred_shape)

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
                sensitivity_im = self._append_to_sensitivity_im(model, batch_images, batch_ids, sensitivity_im)
                batch_images = []
                batch_ids = []

        self._sensitivity_im = sensitivity_im.reshape(output_im_shape)
        self._is_update = True

    def eval(self):
        """
         Computes the occlusion_sensitivity.

         Returns:
             A numpy ndarray.

        """
        if not self._is_update:
            raise RuntimeError('Call the update method before calling eval.')

        sensitivity = self._baseline - np.squeeze(self._sensitivity_im)

        return sensitivity
