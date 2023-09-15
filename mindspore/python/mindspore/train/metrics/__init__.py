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
"""
Metrics.

Functions to measure the performance of the machine learning models
on the evaluation dataset. It's used to choose the best model.
"""
from __future__ import absolute_import

from mindspore.train.metrics.accuracy import Accuracy
from mindspore.train.metrics.hausdorff_distance import HausdorffDistance
from mindspore.train.metrics.error import MAE, MSE
from mindspore.train.metrics.metric import Metric, rearrange_inputs
from mindspore.train.metrics.precision import Precision
from mindspore.train.metrics.recall import Recall
from mindspore.train.metrics.fbeta import Fbeta, F1
from mindspore.train.metrics.dice import Dice
from mindspore.train.metrics.roc import ROC
from mindspore.train.metrics.auc import auc
from mindspore.train.metrics.topk import TopKCategoricalAccuracy, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from mindspore.train.metrics.loss import Loss
from mindspore.train.metrics.mean_surface_distance import MeanSurfaceDistance
from mindspore.train.metrics.root_mean_square_surface_distance import RootMeanSquareDistance
from mindspore.train.metrics.bleu_score import BleuScore
from mindspore.train.metrics.cosine_similarity import CosineSimilarity
from mindspore.train.metrics.occlusion_sensitivity import OcclusionSensitivity
from mindspore.train.metrics.perplexity import Perplexity
from mindspore.train.metrics.confusion_matrix import ConfusionMatrixMetric, ConfusionMatrix

__all__ = [
    "names",
    "get_metric_fn",
    "Accuracy",
    "MAE", "MSE",
    "Metric", "rearrange_inputs",
    "Precision",
    "HausdorffDistance",
    "Recall",
    "Fbeta",
    "BleuScore",
    "CosineSimilarity",
    "OcclusionSensitivity",
    "F1",
    "Dice",
    "ROC",
    "auc",
    "TopKCategoricalAccuracy",
    "Top1CategoricalAccuracy",
    "Top5CategoricalAccuracy",
    "Loss",
    "MeanSurfaceDistance",
    "RootMeanSquareDistance",
    "Perplexity",
    "ConfusionMatrix",
    "ConfusionMatrixMetric",
]

__factory__ = {
    'accuracy': Accuracy,
    'acc': Accuracy,
    'precision': Precision,
    'recall': Recall,
    'F1': F1,
    'dice': Dice,
    'roc': ROC,
    'auc': auc,
    'bleu_score': BleuScore,
    'cosine_similarity': CosineSimilarity,
    'occlusion_sensitivity': OcclusionSensitivity,
    'topk': TopKCategoricalAccuracy,
    'hausdorff_distance': HausdorffDistance,
    'top_1_accuracy': Top1CategoricalAccuracy,
    'top_5_accuracy': Top5CategoricalAccuracy,
    'mae': MAE,
    'mse': MSE,
    'loss': Loss,
    'mean_surface_distance': MeanSurfaceDistance,
    'root_mean_square_distance': RootMeanSquareDistance,
    'perplexity': Perplexity,
    'confusion_matrix': ConfusionMatrix,
    'confusion_matrix_metric': ConfusionMatrixMetric,
}


def names():
    """
    Gets all names of the metric methods.

    Returns:
        List, the name list of metric methods.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> ms.train.names()
        ['F1', 'acc', 'accuracy', 'auc', 'bleu_score', 'confusion_matrix', 'confusion_matrix_metric',
        'cosine_similarity', 'dice', 'hausdorff_distance', 'loss', 'mae', 'mean_surface_distance', 'mse',
        'occlusion_sensitivity', 'perplexity', 'precision', 'recall', 'roc', 'root_mean_square_distance',
        'top_1_accuracy', 'top_5_accuracy', 'topk']
    """
    return sorted(__factory__.keys())


def get_metric_fn(name, *args, **kwargs):
    """
    Gets the metric method based on the input name.

    Args:
        name (str): The name of metric method. Names can be obtained by :func:`mindspore.train.names` .
            object for the currently supported metrics.
        args: Arguments for the metric function.
        kwargs: Keyword arguments for the metric function.

    Returns:
        Metric object, class instance of the metric method.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.train import get_metric_fn
        >>> metric = get_metric_fn('precision', eval_type='classification')
    """
    if name not in __factory__:
        raise KeyError(f"For 'get_metric_fn', unsupported metric {name}, please refer to official website "
                       f"for more details about supported metrics.")
    return __factory__[name](*args, **kwargs)


def get_metrics(metrics):
    """
    Get metrics used in evaluation.

    Args:
        metrics (Union[dict, set]): Dict or set of metrics to be evaluated by the model during training and
                                    testing. eg: {'accuracy', 'recall'}.

    Returns:
        dict, the key is metric name, the value is class instance of metric method.

    Raises:
        TypeError: If the type of argument 'metrics' is not None, dict or set.
    """
    if metrics is None:
        return metrics

    if isinstance(metrics, dict):
        for name, metric in metrics.items():
            if not isinstance(name, str) or not isinstance(metric, Metric):
                raise TypeError(f"For 'get_metrics', if 'metrics' is dict, the key in 'metrics' must be string and "
                                f"value in 'metrics' must be Metric, but got key:{type(name)}, value:{type(metric)}.")
        return metrics
    if isinstance(metrics, set):
        out_metrics = {}
        for name in metrics:
            out_metrics[name] = get_metric_fn(name)
        return out_metrics

    raise TypeError("For 'get_metrics', the argument 'metrics' must be None, dict or set, "
                    "but got {}".format(metrics))
