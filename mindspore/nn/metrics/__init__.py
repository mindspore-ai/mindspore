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
"""
Metrics.

Functions to measure the performance of the machine learning models
on the evaluation dataset. It's used to choose the best model.
"""
from .accuracy import Accuracy
from .error import MAE, MSE
from .metric import Metric
from .precision import Precision
from .recall import Recall
from .fbeta import Fbeta, F1
from .topk import TopKCategoricalAccuracy, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from .loss import Loss

__all__ = [
    "names", "get_metric_fn",
    "Accuracy",
    "MAE", "MSE",
    "Metric",
    "Precision",
    "Recall",
    "Fbeta",
    "F1",
    "TopKCategoricalAccuracy",
    "Top1CategoricalAccuracy",
    "Top5CategoricalAccuracy",
    "Loss",
]

__factory__ = {
    'accuracy': Accuracy,
    'acc': Accuracy,
    'precision': Precision,
    'recall': Recall,
    'F1': F1,
    'topk': TopKCategoricalAccuracy,
    'top_1_accuracy': Top1CategoricalAccuracy,
    'top_5_accuracy': Top5CategoricalAccuracy,
    'mae': MAE,
    'mse': MSE,
    'loss': Loss,
}


def names():
    """
    Gets the names of the metric methods.

    Returns:
        List, the name list of metric methods.
    """
    return sorted(__factory__.keys())


def get_metric_fn(name, *args, **kwargs):
    """
    Gets the metric method based on the input name.

    Args:
        name (str): The name of metric method. Refer to the '__factory__'
            object for the currently supported metrics.
        args: Arguments for the metric function.
        kwargs: Keyword arguments for the metric function.

    Returns:
        Metric object, class instance of the metric method.

    Examples:
        >>> metric = nn.get_metric_fn('precision', eval_type='classification')
    """
    if name not in __factory__:
        raise KeyError("Unknown Metric:", name)
    return __factory__[name](*args, **kwargs)


def get_metrics(metrics):
    """
    Get metrics used in evaluation.

    Args:
        metrics (Union[dict, set]): Dict or set of metrics to be evaluated by the model during training and
                                    testing. eg: {'accuracy', 'recall'}.

    Returns:
        dict, the key is metric name, the value is class instance of metric method.
    """
    if metrics is None:
        return metrics

    if isinstance(metrics, dict):
        for name, metric in metrics.items():
            if not isinstance(name, str) or not isinstance(metric, Metric):
                raise TypeError("Metrics format error. key in metrics should be str \
                                  and value in metrics should be subclass of Metric")
        return metrics
    if isinstance(metrics, set):
        out_metrics = {}
        for name in metrics:
            out_metrics[name] = get_metric_fn(name)
        return out_metrics

    raise TypeError("Metrics should be None, dict or set, but got {}".format(metrics))
