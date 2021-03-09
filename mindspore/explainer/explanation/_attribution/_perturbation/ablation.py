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
"""Modules to ablate images."""

__all__ = [
    'Ablation',
    'AblationWithSaliency',
]

import math
from functools import reduce
from typing import Optional, Union

import numpy as np

from .replacement import Constant
from ...._utils import rank_pixels


class Ablation:
    """Base class to ablate image based on given replacement."""

    def __init__(self, perturb_mode: str):
        self._perturb_mode = perturb_mode

    def __call__(self,
                 inputs: np.array,
                 reference: Union[np.array, float],
                 masks: np.array
                 ) -> np.array:

        """
        Generate perturbations of given array.

        Args:
            inputs (np.ndarray): Input array to perturb. The first dim of inputs is assumed to be the batch size, i.e.,
                number of samples.
            reference (np.ndarray or float): Array of values to replace the elements in the original inputs. The shape
                of reference must match the inputs. If scalar is provided, the perturbed elements will be assigned the
                given value..
            masks (np.ndarray): Several boolean array to mark the perturbed positions. True marks the pixels to be
                perturbed, otherwise the pixels will be kept. The shape of masks is assumed to be
                [batch_size, num_perturbations, inputs_shape[1:]].

        Return:
            perturbations (np.ndarray)
        """
        if isinstance(reference, float):
            reference = Constant(base_value=reference)(inputs)

        if not np.array_equal(inputs.shape, reference.shape):
            raise ValueError('reference must have the same shape as inputs.')

        num_perturbations = masks.shape[1]

        if self._perturb_mode == 'Insertion':
            inputs, reference = reference, inputs

        perturbations = np.repeat(inputs[:, None, :], num_perturbations, 1)
        reference = np.repeat(reference[:, None, :], num_perturbations, 1)
        Ablation._assign(perturbations, reference, masks)

        return perturbations

    @staticmethod
    def _assign(original_array: np.ndarray, replacement: np.ndarray, masks: np.ndarray):
        """Assign values to perturb pixels on perturbations."""
        if masks.dtype != bool:
            raise TypeError('The param "masks" should be an array of bool, but receive {}'.format(masks.dtype))

        if not np.array_equal(original_array.shape, masks.shape):
            raise ValueError('masks must have the shape {} same as [batch_size, num_perturbations, inputs.shape[1:],'
                             'but receive {}.'.format(original_array.shape, masks.shape))

        original_array[masks] = replacement[masks]


class AblationWithSaliency(Ablation):
    """
    Perturbation generator to generate perturbations w.r.t a given saliency map.

    Args:
        perturb_percent (float): percentage of pixels to perturb
        perturb_mode (str): specify perturbing mode, through deleting or
            inserting pixels. Current support: ['Deletion', 'Insertion'].
        is_accumulate (bool): whether to accumulate the former perturbations to
            the later perturbations.
        perturb_pixel_per_step (int, optional): number of pixel to perturb
            for each perturbation. If perturb_pixel_per_step is None, actual
            perturb_pixel_per_step will be calculate by:
                num_image_pixel * perturb_percent / num_perturb_steps.
            Default: None
        num_perturbations (int, optional): number of perturbations. If
            num_perturbations if None, it will be calculated by:
                num_image_pixel * perturb_percent / perturb_pixel_per_step.
            Default: None

     """

    def __init__(self,
                 perturb_mode: str,
                 perturb_percent: float = 1.0,
                 is_accumulate: bool = False,
                 perturb_pixel_per_step: Optional[int] = None,
                 num_perturbations: Optional[int] = None):
        super().__init__(perturb_mode)
        self._perturb_percent = perturb_percent
        self._perturb_mode = perturb_mode
        self._pixel_per_step = perturb_pixel_per_step
        self._num_perturbations = num_perturbations
        self._is_accumulate = is_accumulate

    def generate_mask(self,
                      saliency: np.ndarray,
                      num_channels: Optional[int] = None
                      ) -> np.ndarray:
        """
        Generate mask for perturbations based on given saliency ranks.

        Args:
            saliency (numpy.array): Perturbing masks will be generated based on the given saliency map. The shape of
                saliency is expected to be: [batch_size, optional(num_channels), *spatial_size]. If multi-channel
                saliency is provided, an averaged saliency will be taken to calculate pixel order in spatial dimension.
            num_channels (optional[int]): Number of channels of the input data. In order to match the shape of inputs,
                num_channels should be provided when input data have channels dimension, even if num_channel is 1.
                If None is provided, the inputs is assumed to be no-channel data, and the generated mask will have
                no channel dimension. Default: None.

        Return:
            numpy.array, boolean masks for perturbation generation.
        """

        batch_size = saliency.shape[0]
        has_channel = num_channels is not None
        num_channels = 1 if num_channels is None else num_channels

        if has_channel:
            saliency = saliency.mean(axis=1)
        saliency_rank = rank_pixels(saliency, descending=True)
        num_pixels = reduce(lambda x, y: x * y, saliency.shape[1:])

        pixel_per_step, num_perturbations = self._check_and_format_perturb_param(num_pixels)

        masks = np.zeros((batch_size, num_perturbations, num_channels, saliency_rank.shape[1], saliency_rank.shape[2]),
                         dtype=np.bool)

        # If the perturbation is added accumulately, the factor should be 0 to preserve the low bound of indexing.
        factor = 0 if self._is_accumulate else 1

        for i in range(batch_size):
            low_bound = 0
            up_bound = low_bound + pixel_per_step
            for j in range(num_perturbations):
                masks[i, j, :, ((saliency_rank[i] >= low_bound) & (saliency_rank[i] < up_bound))] = True
                low_bound = up_bound + factor
                up_bound += pixel_per_step

        masks = masks if has_channel else np.squeeze(masks, axis=2)
        return masks

    def _check_and_format_perturb_param(self, num_pixels):
        """
        Check whether the self._pixel_per_step and self._num_perturbation is valid. If the parameters are unreasonable,
        this function will try to reassign the parameters and raise ValueError when reassignment is failed.
        """
        if self._pixel_per_step:
            pixel_per_step = self._pixel_per_step
            num_perturbations = math.floor(num_pixels * self._perturb_percent / self._pixel_per_step)
        elif self._num_perturbations:
            pixel_per_step = math.floor(num_pixels * self._perturb_percent / self._num_perturbations)
            num_perturbations = self._num_perturbations
        else:
            # If neither pixel_per_step or num_perturbations is provided, num_perturbations is determined by the square
            # root of product from the spatial size of saliency map.
            num_perturbations = math.floor(np.sqrt(num_pixels))
            pixel_per_step = math.floor(num_pixels * self._perturb_percent / num_perturbations)

        return pixel_per_step, num_perturbations
