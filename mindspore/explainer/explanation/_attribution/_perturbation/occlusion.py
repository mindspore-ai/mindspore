# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Occlusion explainer."""

import math
from typing import Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import Cell
from .ablation import Ablation
from .perturbation import PerturbationAttribution
from .replacement import Constant
from ...._utils import abs_max

_Array = np.ndarray
_Label = Union[int, Tensor]


def _generate_patches(array, window_size, stride):
    """View as windows."""
    if not isinstance(array, np.ndarray):
        raise TypeError("`array` must be a numpy ndarray")

    arr_shape = np.array(array.shape)
    window_size = np.array(window_size, dtype=arr_shape.dtype)

    slices = tuple(slice(None, None, st) for st in stride)
    window_strides = np.array(array.strides)

    indexing_strides = array[slices].strides
    win_indices_shape = (((np.array(array.shape) - np.array(window_size)) // np.array(stride)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_size))
    strides = tuple(list(indexing_strides) + list(window_strides))

    patches = as_strided(array, shape=new_shape, strides=strides)
    return patches


class Occlusion(PerturbationAttribution):
    r"""
    Occlusion uses a sliding window to replace the pixels with a reference value (e.g. constant value), and computes
    the output difference w.r.t the original output. The output difference caused by perturbed pixels are assigned as
    feature importance to those pixels. For pixels involved in multiple sliding windows, the feature importance is the
    averaged differences from multiple sliding windows.

    For more details, please refer to the original paper via: `<https://arxiv.org/abs/1311.2901>`_.

    Args:
        network (Cell): Specify the black-box model to be explained.

    Inputs:
            inputs (Tensor): The input data to be explained, a 4D tensor of shape :math:`(N, C, H, W)`.
            targets (Tensor, int): The label of interest. It should be a 1D or 0D tensor, or an integer.
                If it is a 1D tensor, its length should be the same as `inputs`.

    Outputs:
            Tensor, a 4D tensor of shape :math:`(N, 1, H, W)`.

    Example:
        >>> from mindspore.explainer.explanation import Occlusion
        >>> net = resnet50(10)
        >>> param_dict = load_checkpoint("resnet50.ckpt")
        >>> load_param_into_net(net, param_dict)
        >>> occlusion = Occlusion(net)
        >>> x = ms.Tensor(np.random.rand([1, 3, 224, 224]), ms.float32)
        >>> label = 1
        >>> saliency = occlusion(x, label)
    """

    def __init__(self, network: Cell, activation_fn: Cell = nn.Softmax()):
        super().__init__(network, activation_fn)

        self._ablation = Ablation(perturb_mode='Deletion')
        self._aggregation_fn = abs_max
        self._get_replacement = Constant(base_value=0.0)
        self._num_sample_per_dim = 32  # specify the number of perturbations each dimension.
        self._num_per_eval = 32  # number of perturbations each evaluation step.

    def __call__(self, inputs: Tensor, targets: _Label) -> Tensor:
        """Call function for 'Occlusion'."""
        self._verify_data(inputs, targets)

        inputs = inputs.asnumpy()
        targets = targets.asnumpy() if isinstance(targets, Tensor) else np.array([targets] * inputs.shape[0], np.int)

        # If spatial size of input data is smaller than self._num_sample_per_dim, window_size and strides will set to
        # `(C, 3, 3)` and `(C, 1, 1)` separately.
        window_size = tuple(
            [inputs.shape[1]]
            + [x % self._num_sample_per_dim if x > self._num_sample_per_dim else 3 for x in inputs.shape[2:]])
        strides = tuple(
            [inputs.shape[1]]
            + [x // self._num_sample_per_dim if x > self._num_sample_per_dim else 1 for x in inputs.shape[2:]])

        model = nn.SequentialCell([self._model, self._activation_fn])

        original_outputs = model(Tensor(inputs, ms.float32)).asnumpy()[np.arange(len(targets)), targets]

        total_attribution = np.zeros_like(inputs)
        weights = np.ones_like(inputs)
        masks = Occlusion._generate_masks(inputs, window_size, strides)
        num_perturbations = masks.shape[1]
        original_outputs_repeat = np.repeat(original_outputs, repeats=num_perturbations, axis=0)

        reference = self._get_replacement(inputs)
        occluded_inputs = self._ablation(inputs, reference, masks)
        targets_repeat = np.repeat(targets, repeats=num_perturbations, axis=0)

        occluded_inputs = occluded_inputs.reshape((-1, *inputs.shape[1:]))
        if occluded_inputs.shape[0] > self._num_per_eval:
            cal_time = math.ceil(occluded_inputs.shape[0] / self._num_per_eval)
            occluded_outputs = []
            for i in range(cal_time):
                occluded_input = occluded_inputs[i*self._num_per_eval
                                                 :min((i+1) * self._num_per_eval, occluded_inputs.shape[0])]
                target = targets_repeat[i*self._num_per_eval
                                        :min((i+1) * self._num_per_eval, occluded_inputs.shape[0])]
                occluded_output = model(Tensor(occluded_input)).asnumpy()[np.arange(target.shape[0]), target]
                occluded_outputs.append(occluded_output)
            occluded_outputs = np.concatenate(occluded_outputs)
        else:
            occluded_outputs = model(Tensor(occluded_inputs)).asnumpy()[np.arange(len(targets_repeat)), targets_repeat]
        outputs_diff = original_outputs_repeat - occluded_outputs
        outputs_diff = outputs_diff.reshape(inputs.shape[0], -1)

        total_attribution += (
            outputs_diff.reshape(outputs_diff.shape + (1,) * (len(masks.shape) - 2)) * masks).sum(axis=1).clip(1e-6)
        weights += masks.sum(axis=1)

        attribution = self._aggregation_fn(ms.Tensor(total_attribution / weights))
        return attribution

    @staticmethod
    def _generate_masks(inputs: Tensor, window_size: Tuple[int, ...], strides: Tuple[int, ...]) -> _Array:
        """Generate masks to perturb contiguous regions."""
        total_dim = np.prod(inputs.shape[1:]).item()
        template = np.arange(total_dim).reshape(inputs.shape[1:])
        indices = _generate_patches(template, window_size, strides)
        num_perturbations = indices.reshape((-1,) + window_size).shape[0]
        indices = indices.reshape(num_perturbations, -1)

        mask = np.zeros((num_perturbations, total_dim), dtype=np.bool)
        for i in range(num_perturbations):
            mask[i, indices[i]] = True
        mask = mask.reshape((num_perturbations,) + inputs.shape[1:])

        masks = np.tile(mask, reps=(inputs.shape[0],) + (1,) * len(mask.shape))
        return masks
