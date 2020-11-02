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
"""Faithfulness"""
import math
from typing import Callable, Optional, Union, Tuple

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from mindspore import log
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as op
from .metric import AttributionMetric
from ..._utils import calc_correlation, calc_auc, format_tensor_to_ndarray, rank_pixels
from ...explanation._attribution._attribution import Attribution as _Attribution

_Array = np.ndarray
_Explainer = Union[_Attribution, Callable]
_Label = Union[int, ms.Tensor]
_Module = nn.Cell


def _calc_feature_importance(saliency: _Array, masks: _Array) -> _Array:
    """Calculate feature important w.r.t given masks."""
    feature_importance = []
    num_perturbations = masks.shape[0]
    for i in range(num_perturbations):
        patch_feature_importance = saliency[masks[i]].sum() / masks[i].sum()
        feature_importance.append(patch_feature_importance)
    feature_importance = np.array(feature_importance, dtype=np.float32)
    return feature_importance


class _BaseReplacement:
    """
    Base class of generator for generating different replacement for perturbations.

    Args:
        kwargs: Optional args for generating replacement. Derived class need to
            add necessary arg names and default value to '_necessary_args'.
            If the argument has no default value, the value should be set to
            'EMPTY' to mark the required args. Initializing an object will
            check the given kwargs w.r.t '_necessary_args'.

    Raise:
        ValueError: Raise when provided kwargs not contain necessary arg names with 'EMPTY' mark.
    """
    _necessary_args = {}

    def __init__(self, **kwargs):
        self._replace_args = self._necessary_args.copy()
        for key, value in self._replace_args.items():
            if key in kwargs.keys():
                self._replace_args[key] = kwargs[key]
            elif key not in kwargs.keys() and value == 'EMPTY':
                raise ValueError(f"Missing keyword arg {key} for {self.__class__.__name__}.")

    __call__: Callable
    """
    Generate replacement for perturbations. Derived class should overwrite this
    function to generate different replacement for perturbing.

    Args:
        inputs (_Array): Array to be perturb.

    Returns:
        - replacement (_Array): Array to provide alternative pixels for every
        position in the given
            inputs. The returned array should have same shape as inputs.
    """


class Constant(_BaseReplacement):
    """ Generator to provide constant-value replacement for perturbations """
    _necessary_args = {'base_value': 'EMPTY'}

    def __call__(self, inputs: _Array) -> _Array:
        replacement = np.ones_like(inputs, dtype=np.float32)
        replacement *= self._replace_args['base_value']
        return replacement


class GaussianBlur(_BaseReplacement):
    """ Generator to provided gaussian blurred inputs for perturbation. """
    _necessary_args = {'sigma': 0.7}

    def __call__(self, inputs: _Array) -> _Array:
        sigma = self._replace_args['sigma']
        replacement = gaussian_filter(inputs, sigma=sigma)
        return replacement


class Perturb:
    """
    Perturbation generator to generate perturbations for a given array.

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
                 perturb_percent: float,
                 perturb_mode: str,
                 is_accumulate: bool,
                 perturb_pixel_per_step: Optional[int] = None,
                 num_perturbations: Optional[int] = None):
        self._perturb_percent = perturb_percent
        self._perturb_mode = perturb_mode
        self._pixel_per_step = perturb_pixel_per_step
        self._num_perturbations = num_perturbations
        self._is_accumulate = is_accumulate

    @staticmethod
    def _assign(x: _Array, y: _Array, masks: _Array):
        """Assign values to perturb pixels on perturbations."""
        if masks.dtype != bool:
            raise TypeError('The param "masks" should be an array of bool, but receive {}'
                            .format(masks.dtype))
        for i in range(x.shape[0]):
            x[i][:, masks[i]] = y[:, masks[i]]

    def _generate_mask(self, saliency_rank: _Array) -> _Array:
        """Generate mask for perturbations based on given saliency ranks."""
        if len(saliency_rank.shape) != 2:
            raise ValueError(f'The param "saliency_rank" should be 2-dim, but receive {len(saliency_rank.shape)}.')

        num_pixels = saliency_rank.shape[0] * saliency_rank.shape[1]
        if self._pixel_per_step:
            pixel_per_step = self._pixel_per_step
            num_perturbations = math.floor(
                num_pixels * self._perturb_percent / self._pixel_per_step)
        elif self._num_perturbations:
            pixel_per_step = math.floor(
                num_pixels * self._perturb_percent / self._num_perturbations)
            num_perturbations = self._num_perturbations
        else:
            raise ValueError("Must provide either pixel_per_step or num_perturbations.")

        masks = np.zeros(
            (num_perturbations, saliency_rank.shape[0], saliency_rank.shape[1]),
            dtype=np.bool)
        low_bound = 0
        up_bound = low_bound + pixel_per_step
        factor = 0 if self._is_accumulate else 1

        for i in range(num_perturbations):
            masks[i, ((saliency_rank >= low_bound)
                      & (saliency_rank < up_bound))] = True
            low_bound = up_bound * factor
            up_bound += pixel_per_step

        if len(masks.shape) == 3:
            return masks
        raise ValueError(f'Invalid masks shape {len(masks.shape)}, expect 3-dim.')

    def __call__(self,
                 inputs: _Array,
                 saliency: _Array,
                 reference: _Array,
                 return_mask: bool = False,
                 ) -> Union[_Array, Tuple[_Array, ...]]:
        """
        Generate perturbations of given array.

        Args:
            inputs (_Array): input array to perturb
            saliency (_Array): saliency map
            return_mask (bool): whether return the mask for generating
                the perturbation. The mask can be used to calculate
                average feature importance of pixels perturbed at each step.

        Return:
            perturbations (_Array)
            masks (_Array): return when return_mask is set to True.
        """
        if not np.array_equal(inputs.shape, reference.shape):
            raise ValueError('reference must have the same shape as inputs.')

        saliency_rank = rank_pixels(saliency, descending=True)
        masks = self._generate_mask(saliency_rank)
        num_perturbations = masks.shape[0]

        if self._perturb_mode == 'Insertion':
            inputs, reference = reference, inputs

        perturbations = np.tile(
            inputs, (num_perturbations, *[1] * len(inputs.shape)))

        Perturb._assign(perturbations, reference, masks)

        if return_mask:
            return perturbations, masks
        return perturbations


class _FaithfulnessHelper:
    """Base class for faithfulness calculator."""
    _support = [Constant, GaussianBlur]

    def __init__(self,
                 perturb_percent: float,
                 perturb_mode: str,
                 perturb_method: str,
                 is_accumulate: bool,
                 perturb_pixel_per_step: Optional[int] = None,
                 num_perturbations: Optional[int] = None,
                 **kwargs):

        self._get_reference = None
        for method in self._support:
            if perturb_method == method.__name__:
                self._get_reference = method(**kwargs)
        if self._get_reference is None:
            raise ValueError(
                'The param "perturb_method" should be one of {}.'.format([x.__name__ for x in self._support]))

        self._perturb = Perturb(perturb_percent=perturb_percent,
                                perturb_mode=perturb_mode,
                                perturb_pixel_per_step=perturb_pixel_per_step,
                                num_perturbations=num_perturbations,
                                is_accumulate=is_accumulate)

    calc_faithfulness: Callable
    """
    Method used to calculate faithfulness for given inputs, target label,
    saliency. Derive class should implement this method.

    Args:
        inputs (_Array): sample to calculate faithfulness score
        model (_Module): model to explanation
        targets (_Label): label to explanation on.
        saliency (_Array): Saliency map of given inputs and targets from the
            explainer.

    Return:
        - faithfulness (float): faithfulness score
    """


class NaiveFaithfulness(_FaithfulnessHelper):
    """
    Calculator for naive faithfulness.

    Naive faithfulness, the metric replace several pixels on original image by
    specific method for each perturbations. The metric predicts on the perturbed
    images and record a series of probabilities. Then calculates the
    correlation between prob distribution and averaged feature importance.
    Higher correlation indicates better faithfulness.

    Args:
        perturb_percent (float): percentage of pixels to perturb
        perturb_method (str): specify the method to replace the pixel.
            Current support: ['Constant', 'GaussianBlur']
        is_accumulate (bool): whether to accumulate the former perturbations to
            the later perturbations.
            Default: False.
        perturb_pixel_per_step (Optional[int]): number of pixel to perturb
            for each perturbation. If perturb_pixel_per_step is None, actual
            perturb_pixel_per_step will be calculate by:
                num_image_pixel * perturb_percent / num_perturb_steps.
            Default: None
        num_perturbations (Optional[int]): number of perturbations. If
            num_perturbations if None, it will be calculated by:
                num_image_pixel * perturb_percent / perturb_pixel_per_step.
            Default: None
        kwargs: specific perturb_method will require
            different arguments. Below lists required args for each method.

            'Constant': base_value (int)
            'GaussianBlur': sigma (float): 0.7

    """

    def __init__(self,
                 perturb_percent: float,
                 perturb_method: str,
                 is_accumulate: bool = False,
                 perturb_pixel_per_step: Optional[int] = None,
                 num_perturbations: Optional[int] = None,
                 **kwargs):
        super(NaiveFaithfulness, self).__init__(
            perturb_percent=perturb_percent,
            perturb_mode='Deletion',
            perturb_method=perturb_method,
            is_accumulate=is_accumulate,
            perturb_pixel_per_step=perturb_pixel_per_step,
            num_perturbations=num_perturbations,
            **kwargs)

    def calc_faithfulness(self,
                          inputs: _Array,
                          model: _Module,
                          targets: _Label,
                          saliency: _Array) -> np.ndarray:
        """
        Calculate naive faithfulness.

        Args:
            inputs (_Array): sample to calculate faithfulness score
            model (_Module): model to explanation
            targets (_Label): label to explanation on.
            saliency (_Array): Saliency map of given inputs and targets from the
                explainer.

        Return:
            - faithfulness (np.ndarray): faithfulness score

        """
        if not np.count_nonzero(saliency):
            log.warning("The saliency map is zero everywhere. The correlation will be set to zero.")
            correlation = 0
            normalized_faithfulness = (correlation + 1) / 2
            return np.array([normalized_faithfulness], np.float)
        reference = self._get_reference(inputs)
        perturbations, masks = self._perturb(
            inputs, saliency, reference, return_mask=True)
        feature_importance = _calc_feature_importance(saliency, masks)

        perturbations = ms.Tensor(perturbations, dtype=ms.float32)
        predictions = model(perturbations).asnumpy()[:, targets]

        faithfulness = calc_correlation(feature_importance, predictions)
        normalized_faithfulness = (faithfulness + 1) / 2
        return np.array([normalized_faithfulness], np.float)


class DeletionAUC(_FaithfulnessHelper):
    """ Calculator for deletion AUC.

    For Deletion AUC, the metric accumulative replace pixels on origin
    images through specific 'perturb_method', predict on the perturbed images
    and record series of probabilities. The metric then calculates the AUC of
    the probability variation curve during perturbations. Faithfulness is define
    as (1 - deletion_AUC). Higher score indicates better faithfulness of
    explanation.

    Args:
        perturb_percent (float): percentage of pixels to perturb
        perturb_method (str): specify the method to replace the pixel.
            Current support: ['Constant', 'GaussianBlur']
        perturb_pixel_per_step (Optional[int]): number of pixel to perturb
            for each perturbation. If perturb_pixel_per_step is None, actual
            perturb_pixel_per_step will be calculate by:
                num_image_pixel * perturb_percent / num_perturb_steps.
            Default: None
        num_perturbations (Optional[int]): number of perturbations. If
            num_perturbations if None, it will be calculated by:
                num_image_pixel * perterb_percent / perturb_pixel_per_step.
            Default: None
        kwargs: specific perturb_method will require
            different arguments. Below lists required args for each method.

            'Constant': base_value (int)
            'GaussianBlur': sigma (float): 0.7

    """

    def __init__(self,
                 perturb_percent: float,
                 perturb_method: str,
                 perturb_pixel_per_step: Optional[int] = None,
                 num_perturbations: Optional[int] = None,
                 **kwargs):
        super(DeletionAUC, self).__init__(
            perturb_percent=perturb_percent,
            perturb_mode='Deletion',
            perturb_method=perturb_method,
            perturb_pixel_per_step=perturb_pixel_per_step,
            num_perturbations=num_perturbations,
            is_accumulate=True,
            **kwargs)

    def calc_faithfulness(self,
                          inputs: _Array,
                          model: _Module,
                          targets: _Label,
                          saliency: _Array) -> np.ndarray:
        """
        Calculate faithfulness through deletion AUC.

        Args:
            inputs (_Array): sample to calculate faithfulness score
            model (_Module): model to explanation
            targets (_Label): label to explanation on.
            saliency (_Array): Saliency map of given inputs and targets from the
                explainer.

        Return:
            - faithfulness (float): faithfulness score

        """
        reference = self._get_reference(inputs)
        perturbations = self._perturb(inputs, saliency, reference)
        perturbations = ms.Tensor(perturbations, dtype=ms.float32)
        predictions = model(perturbations).asnumpy()[:, targets]
        input_tensor = op.ExpandDims()(ms.Tensor(inputs, ms.float32), 0)
        original_output = model(input_tensor).asnumpy()[:, targets]

        auc = calc_auc(original_output - predictions)
        return np.array([1 - auc])


class InsertionAUC(_FaithfulnessHelper):
    """ Calculator for insertion AUC.

    For Insertion AUC, the metric accumulative replace pixels of reference
    image by pixels from origin image, like inserting pixel from origin image to
    reference. The reference if generated through specific 'perturb_method'.
    The metric predicts on the perturbed images and records series of
    probabilities. The metric then calculates the AUC of the probability
    variation curve during perturbations. Faithfulness is define as (1 -
    deletion_AUC). Higher score indicates better faithfulness of explanation.

    Args:
        perturb_percent (float): percentage of pixels to perturb
        perturb_method (str): specify the method to replace the pixel.
            Current support: ['Constant', 'GaussianBlur']
        perturb_pixel_per_step (Optional[int]): number of pixel to perturb
            for each perturbation. If perturb_pixel_per_step is None, actual
            perturb_pixel_per_step will be calculate by:
                num_image_pixel * perturb_percent / num_perturb_steps.
            Default: None
        num_perturbations (Optional[int]): number of perturbations. If
            num_perturbations if None, it will be calculated by:
                num_image_pixel * perterb_percent / perturb_pixel_per_step.
            Default: None
        kwargs: specific perturb_method will require
            different arguments. Below lists required args for each method.

            'Constant': base_value (int)
            'GaussianBlur': sigma (float): 0.7

    """

    def __init__(self,
                 perturb_percent: float,
                 perturb_method: str,
                 perturb_pixel_per_step: Optional[int] = None,
                 num_perturbations: Optional[int] = None,
                 **kwargs):
        super(InsertionAUC, self).__init__(
            perturb_percent=perturb_percent,
            perturb_mode='Insertion',
            perturb_method=perturb_method,
            perturb_pixel_per_step=perturb_pixel_per_step,
            num_perturbations=num_perturbations,
            is_accumulate=True,
            **kwargs)

    def calc_faithfulness(self,
                          inputs: _Array,
                          model: _Module,
                          targets: _Label,
                          saliency: _Array) -> np.ndarray:
        """
        Calculate faithfulness through insertion AUC.

        Args:
            inputs (_Array): sample to calculate faithfulness score
            model (_Module): model to explanation
            targets (_Label): label to explanation on.
            saliency (_Array): Saliency map of given inputs and targets from the
                explainer.

        Return:
            - faithfulness (float): faithfulness score

        """
        reference = self._get_reference(inputs)
        perturbations = self._perturb(inputs, saliency, reference)
        perturbations = ms.Tensor(perturbations, dtype=ms.float32)
        predictions = model(perturbations).asnumpy()[:, targets]
        base_tensor = op.ExpandDims()(ms.Tensor(reference, ms.float32), 0)
        base_outputs = model(base_tensor).asnumpy()[:, targets]

        auc = calc_auc(predictions - base_outputs)
        return np.array([auc])


class Faithfulness(AttributionMetric):
    """
    Provides evaluation on faithfulness on XAI explanations.

    Three specific metrics to obtain quantified results are supported: "NaiveFaithfulness", "DeletionAUC", and
    "InsertionAUC".

    For metric "NaiveFaithfulness", a series of perturbed images are created by modifying pixels
    on original image. Then the perturbed images will be fed to the model and a series of output probability drops can
    be obtained. The faithfulness is then quantified as the correlation between the propability drops and the saliency
    map values on the same pixels (we normalize the correlation further to make them in range of [0, 1]).

    For metric "DeletionAUC", a series of perturbed images are created by accumulatively modifying pixels of the
    original image to a base value (e.g. a constant). The perturbation starts from pixels with high saliency values
    to pixels with low saliency values. Feeding the perturbed images into the model in order, an output probability
    drop curve can be obtained. "DeletionAUC" is then obtained as the area under this probability drop curve.

    For metric "InsertionAUC", a series of perturbed images are created by accumulatively inserting pixels of the
    original image to a reference image (e.g. a black image). The insertion starts from pixels with high saliency values
    to pixels with low saliency values. Feeding the perturbed images into the model in order, an output probability
    increase curve can be obtained. "InsertionAUC" is then obtained as the area under this curve.

    For all the three metrics, higher value indicates better faithfulness.

    Args:
        num_labels (int): Number of labels.
        metric (str, optional): The specifi metric to quantify faithfulness.
            Options: "DeletionAUC", "InsertionAUC", "NaiveFaithfulness".
            Default: 'NaiveFaithfulness'.

    Examples:
        >>> from mindspore.explainer.benchmark import Faithfulness
        >>> # init a `Faithfulness` object
        >>> num_labels = 10
        >>> metric = "InsertionAUC"
        >>> faithfulness = Faithfulness(num_labels, metric)
    """
    _methods = [NaiveFaithfulness, DeletionAUC, InsertionAUC]

    def __init__(self, num_labels: int, metric: str = "NaiveFaithfulness"):
        super(Faithfulness, self).__init__(num_labels)

        perturb_percent = 0.5  # ratio of pixels to be perturbed, future argument
        perturb_method = "Constant"  # perturbation method, all the perturbed pixels will be set to constant
        num_perturb_pixel_per_step = None  # number of pixels for each perturbation step
        num_perturb_steps = 100  # separate the perturbation progress in to 100 steps.
        base_value = 0.0  # the pixel value set for the perturbed pixels

        self._verify_metrics(metric)
        for method in self._methods:
            if metric == method.__name__:
                self._faithfulness_helper = method(
                    perturb_percent=perturb_percent,
                    perturb_method=perturb_method,
                    perturb_pixel_per_step=num_perturb_pixel_per_step,
                    num_perturbations=num_perturb_steps,
                    base_value=base_value
                )

    def evaluate(self, explainer, inputs, targets, saliency=None):
        """
        Evaluate faithfulness on a single data sample.

        Note:
            To apply `Faithfulness` to evaluate an explainer, this explainer must be initialized with a network that
            contains the output activation function. Otherwise, the results will not be correct. Currently only single
            sample (:math:`N=1`) at each call is supported.

        Args:
            explainer (Explanation): The explainer to be evaluated, see `mindspore.explainer.explanation`.
            inputs (Tensor): A data sample, a 4D tensor of shape :math:`(N, C, H, W)`.
            targets (Tensor, int): The label of interest. It should be a 1D or 0D tensor, or an integer.
                If `targets` is a 1D tensor, its length should be the same as `inputs`.
            saliency (Tensor, optional): The saliency map to be evaluated, a 4D tensor of shape :math:`(N, 1, H, W)`.
                If it is None, the parsed `explainer` will generate the saliency map with `inputs` and `targets` and
                continue the evaluation. Default: None.

        Returns:
            numpy.ndarray, 1D array of shape :math:`(N,)`, result of faithfulness evaluated on `explainer`.

        Examples:
            >>> # init an explainer, the network should contain the output activation function.
            >>> network = nn.SequentialCell([resnet50, nn.Sigmoid()])
            >>> gradient = Gradient(network)
            >>> inputs = ms.Tensor(np.random.rand(1, 3, 224, 224), ms.float32)
            >>> targets = 5
            >>> # usage 1: input the explainer and the data to be explained,
            >>> # calculate the faithfulness with the specified metric
            >>> res = faithfulness.evaluate(gradient, inputs, targets)
            >>> # usage 2: input the generated saliency map
            >>> saliency = gradient(inputs, targets)
            >>> res = faithfulenss.evaluate(gradient, inputs, targets, saliency)
        """

        self._check_evaluate_param(explainer, inputs, targets, saliency)

        if saliency is None:
            saliency = explainer(inputs, targets)

        inputs = format_tensor_to_ndarray(inputs)
        saliency = format_tensor_to_ndarray(saliency)

        inputs = inputs.squeeze(axis=0)
        saliency = saliency.squeeze()
        if len(saliency.shape) != 2:
            raise ValueError('Squeezed saliency map is expected to 2D, but receive {}.'.format(len(saliency.shape)))

        faithfulness = self._faithfulness_helper.calc_faithfulness(inputs=inputs, model=explainer.model,
                                                                   targets=targets, saliency=saliency)
        return faithfulness

    def _verify_metrics(self, metric: str):
        supports = [x.__name__ for x in self._methods]
        if metric not in supports:
            raise ValueError("Metric should be one of {}.".format(supports))
