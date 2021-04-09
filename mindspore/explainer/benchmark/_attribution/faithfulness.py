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
"""Faithfulness."""
from decimal import Decimal
from typing import Callable, Optional, Union

import numpy as np

import mindspore as ms
from mindspore import log, nn
from mindspore.train._utils import check_value_type
from .metric import LabelSensitiveMetric
from ..._utils import calc_auc, format_tensor_to_ndarray
from ...explanation._attribution import Attribution as _Attribution
from ...explanation._attribution._perturbation.replacement import Constant, GaussianBlur
from ...explanation._attribution._perturbation.ablation import AblationWithSaliency

_Array = np.ndarray
_Explainer = Union[_Attribution, Callable]
_Label = Union[int, ms.Tensor]
_Module = nn.Cell


def _calc_feature_importance(saliency: _Array, masks: _Array) -> _Array:
    """Calculate feature important w.r.t given masks."""
    if saliency.shape[1] < masks.shape[2]:
        saliency = np.repeat(saliency, repeats=masks.shape[2], axis=1)

    batch_size = masks.shape[0]
    num_perturbations = masks.shape[1]
    saliency = np.repeat(saliency, repeats=num_perturbations, axis=0)
    saliency = saliency.reshape([batch_size, num_perturbations, -1])
    masks = masks.reshape([batch_size, num_perturbations, -1])
    feature_importance = saliency * masks
    feature_importance = feature_importance.sum(-1) / masks.sum(-1)
    return feature_importance


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

        self._ablation = AblationWithSaliency(perturb_mode=perturb_mode,
                                              perturb_percent=perturb_percent,
                                              perturb_pixel_per_step=perturb_pixel_per_step,
                                              num_perturbations=num_perturbations,
                                              is_accumulate=is_accumulate)

    def calc_faithfulness(self, inputs, model, targets, saliency):
        """Calc faithfulness."""
        raise NotImplementedError


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
        super().__init__(perturb_percent=perturb_percent,
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
        if Decimal(str(saliency.max())) == Decimal(str(saliency.min())):
            log.warning("The saliency map is uniform everywhere. The correlation will be set to zero.")
            correlation = 0
            return np.array([correlation], np.float)

        batch_size = inputs.shape[0]
        reference = self._get_reference(inputs)
        masks = self._ablation.generate_mask(saliency, inputs.shape[1])
        perturbations = self._ablation(inputs, reference, masks)
        feature_importance = _calc_feature_importance(saliency, masks)

        perturbations = perturbations.reshape(-1, *perturbations.shape[2:])
        perturbations = ms.Tensor(perturbations, dtype=ms.float32)
        predictions = model(perturbations)[:, targets].asnumpy()
        predictions = predictions.reshape(*feature_importance.shape)

        if Decimal(str(predictions.max())) == Decimal(str(predictions.min())):
            log.warning("The perturbations do not affect the predictions. The correlation will be set to zero.")
            correlation = 0
            return np.array([correlation], np.float)

        faithfulness = -np.corrcoef(feature_importance, predictions)
        faithfulness = np.diag(faithfulness[:batch_size, batch_size:])
        return faithfulness


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
        super().__init__(perturb_percent=perturb_percent,
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
                          saliency: _Array) -> _Array:
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
        masks = self._ablation.generate_mask(saliency, inputs.shape[1])
        perturbations = self._ablation(inputs, reference, masks)
        perturbations = perturbations.reshape(-1, *perturbations.shape[2:])
        perturbations = ms.Tensor(perturbations, dtype=ms.float32)
        predictions = model(perturbations).asnumpy()[:, targets]
        predictions = predictions.reshape((inputs.shape[0], -1))
        input_tensor = ms.Tensor(inputs, ms.float32)
        original_output = model(input_tensor).asnumpy()[:, targets]

        auc = calc_auc(original_output.squeeze() - predictions.squeeze())
        return np.array([1 - auc], np.float)


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
        super().__init__(perturb_percent=perturb_percent,
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
                          saliency: _Array) -> _Array:
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
        masks = self._ablation.generate_mask(saliency, inputs.shape[1])
        perturbations = self._ablation(inputs, reference, masks)
        perturbations = perturbations.reshape(-1, *perturbations.shape[2:])
        perturbations = ms.Tensor(perturbations, dtype=ms.float32)
        predictions = model(perturbations).asnumpy()[:, targets]
        predictions = predictions.reshape((inputs.shape[0], -1))

        base_tensor = ms.Tensor(reference, ms.float32)
        base_outputs = model(base_tensor).asnumpy()[:, targets]

        auc = calc_auc(predictions.squeeze() - base_outputs.squeeze())
        return np.array([auc], np.float)


class Faithfulness(LabelSensitiveMetric):
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
        activation_fn (Cell): The activation layer that transforms logits to prediction probabilities. For
            single label classification tasks, `nn.Softmax` is usually applied. As for multi-label classification tasks,
            `nn.Sigmoid` is usually be applied. Users can also pass their own customized `activation_fn` as long as
            when combining this function with network, the final output is the probability of the input.
        metric (str, optional): The specifi metric to quantify faithfulness.
            Options: "DeletionAUC", "InsertionAUC", "NaiveFaithfulness".
            Default: 'NaiveFaithfulness'.
    """
    _methods = [NaiveFaithfulness, DeletionAUC, InsertionAUC]

    def __init__(self, num_labels, activation_fn, metric="NaiveFaithfulness"):
        super(Faithfulness, self).__init__(num_labels)

        perturb_percent = 0.5  # ratio of pixels to be perturbed, future argument
        perturb_method = "Constant"  # perturbation method, all the perturbed pixels will be set to constant
        base_value = 0.0  # the pixel value set for the perturbed pixels

        check_value_type("activation_fn", activation_fn, nn.Cell)
        self._activation_fn = activation_fn

        self._verify_metrics(metric)
        for method in self._methods:
            if metric == method.__name__:
                self._faithfulness_helper = method(
                    perturb_percent=perturb_percent,
                    perturb_method=perturb_method,
                    base_value=base_value
                )

    def evaluate(self, explainer, inputs, targets, saliency=None):
        """
        Evaluate faithfulness on a single data sample.

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

        Returns:
            numpy.ndarray, 1D array of shape :math:`(N,)`, result of faithfulness evaluated on `explainer`.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> from mindspore.explainer.benchmark import Faithfulness
            >>> from mindspore.explainer.explanation import Gradient
            >>>
            >>> # init a `Faithfulness` object
            >>> num_labels = 10
            >>> metric = "InsertionAUC"
            >>> activation_fn = nn.Softmax()
            >>> faithfulness = Faithfulness(num_labels, activation_fn, metric)
            >>> # The detail of LeNet5 is shown in model_zoo.official.cv.lenet.src.lenet.py
            >>> net = LeNet5(10, num_channel=3)
            >>> gradient = Gradient(net)
            >>> inputs = ms.Tensor(np.random.rand(1, 3, 32, 32), ms.float32)
            >>> targets = 5
            >>> # usage 1: input the explainer and the data to be explained,
            >>> # faithfulness is a Faithfulness instance
            >>> res = faithfulness.evaluate(gradient, inputs, targets)
            >>> print(res.shape)
            (1,)
            >>> # usage 2: input the generated saliency map
            >>> saliency = gradient(inputs, targets)
            >>> res = faithfulness.evaluate(gradient, inputs, targets, saliency)
            >>> print(res.shape)
            (1,)
        """

        self._check_evaluate_param(explainer, inputs, targets, saliency)

        if saliency is None:
            saliency = explainer(inputs, targets)

        inputs = format_tensor_to_ndarray(inputs)
        saliency = format_tensor_to_ndarray(saliency)

        full_network = nn.SequentialCell([explainer.network, self._activation_fn])
        faithfulness = self._faithfulness_helper.calc_faithfulness(inputs=inputs, model=full_network,
                                                                   targets=targets, saliency=saliency)
        return (1 + faithfulness) / 2

    def _verify_metrics(self, metric: str):
        supports = [x.__name__ for x in self._methods]
        if metric not in supports:
            raise ValueError("Metric should be one of {}.".format(supports))
