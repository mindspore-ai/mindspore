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
"""ConfusionMatrixMetric & ConfusionMatrix."""
from __future__ import absolute_import

import numpy as np

from mindspore import _checkparam as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs


class ConfusionMatrix(Metric):
    """
    Computes the confusion matrix, which is commonly used to evaluate the performance of classification models,
    including binary classification and multiple classification.

    If you only need confusion matrix, use this class. If you want to calculate other metrics, such as 'PPV',
    'TPR', 'TNR', etc., use class :class:`mindspore.train.ConfusionMatrixMetric` .

    Args:
        num_classes (int): Number of classes in the dataset.
        normalize (str): Normalization mode for confusion matrix. Default: ``"no_norm"`` . Choose from:

            - **"no_norm"** (None) - No Normalization is used. Default: ``None``.
            - **"target"** (str) - Normalization based on target value.
            - **"prediction"** (str) - Normalization based on predicted value.
            - **"all"** (str) - Normalization over the whole matrix.

        threshold (float): The threshold used to compare with the input tensor. Default: ``0.5`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import ConfusionMatrix
        >>>
        >>> x = Tensor(np.array([1, 0, 1, 0]))
        >>> y = Tensor(np.array([1, 0, 0, 1]))
        >>> metric = ConfusionMatrix(num_classes=2, normalize='no_norm', threshold=0.5)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> output = metric.eval()
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
    """
    def __init__(self, num_classes, normalize="no_norm", threshold=0.5):
        super(ConfusionMatrix, self).__init__()

        self.num_classes = validator.check_value_type("num_classes", num_classes, [int])
        if normalize not in ["target", "prediction", "all", "no_norm"]:
            raise ValueError("For 'ConfusionMatrix', the argument 'normalize' must be in "
                             "['all', 'prediction', 'label', 'no_norm'(None)], but got {}.".format(normalize))

        self.normalize = normalize
        self.threshold = validator.check_value_type("threshold", threshold, [float])
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self._is_update = False

    @rearrange_inputs
    def update(self, *inputs):
        """
        Update state with y_pred and y.

        Args:
            inputs(tuple): Input `y_pred` and `y`. `y_pred` and `y` are a `Tensor`, list or numpy.ndarray.
                    `y_pred` is the predicted value, `y` is the true value.
                    The shape of `y_pred` is :math:`(N, C, ...)` or :math:`(N, ...)`.
                    The shape of `y` is :math:`(N, ...)`.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If the dim of y_pred and y are not equal.
        """
        if len(inputs) != 2:
            raise ValueError("For 'ConfusionMatrix.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        if not (y_pred.ndim == y.ndim or y_pred.ndim == y.ndim + 1):
            raise ValueError(f"For 'ConfusionMatrix.update', predicted value (input[0]) and true value "
                             f"(input[1]) should have same dimensions, or the dimension of predicted value "
                             f"equals the dimension of true value add 1, but got predicted value ndim: "
                             f"{y_pred.ndim}, true value ndim: {y.ndim}.")

        if y_pred.ndim == y.ndim + 1:
            y_pred = np.argmax(y_pred, axis=1)

        if y_pred.ndim == y.ndim and y_pred.dtype in (np.float16, np.float32, np.float64):
            y_pred = (y_pred >= self.threshold).astype(int)

        trans = (y.reshape(-1) * self.num_classes + y_pred.reshape(-1)).astype(int)
        bincount = np.bincount(trans, minlength=self.num_classes ** 2)
        confusion_matrix = bincount.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion_matrix
        self._is_update = True

    def eval(self):
        """
        Computes confusion matrix.

        Returns:
            numpy.ndarray, the computed result.
        """

        if not self._is_update:
            raise RuntimeError("Please call the 'update' method before calling 'eval' method.")

        confusion_matrix = self.confusion_matrix.astype(float)

        matrix_target = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        matrix_pred = confusion_matrix / confusion_matrix.sum(axis=0, keepdims=True)
        matrix_all = confusion_matrix / confusion_matrix.sum()
        normalize_dict = {"target": matrix_target,
                          "prediction": matrix_pred,
                          "all": matrix_all}

        if self.normalize == "no_norm":
            return confusion_matrix

        matrix = normalize_dict.get(self.normalize)
        if matrix[np.isnan(matrix)].size != 0:
            matrix[np.isnan(matrix)] = 0

        return matrix


class ConfusionMatrixMetric(Metric):
    r"""
    Computes metrics related to confusion matrix. The calculation based on full-scale tensor, average values of
    batch, class channel and iteration are collected. All metrics supported by the interface are listed in comments
    of `metric_name`.

    If you want to calculate metrics related to confusion matrix, such as 'PPV', 'TPR', 'TNR', use this class.
    If you only want to calculate confusion matrix, please use :class:`mindspore.train.ConfusionMatrix` .

    Args:
        skip_channel (bool): Whether to skip the measurement calculation on the first channel of the predicted output.
                             Default: ``True`` .
        metric_name (str): Names of supported metrics , users can also set the industry common aliases for them. Choose
                           from: ["sensitivity", "specificity", "precision", "negative predictive value", "miss rate",
                           "fall out", "false discovery rate", "false omission rate", "prevalence threshold",
                           "threat score", "accuracy", "balanced accuracy", "f1 score",
                           "matthews correlation coefficient", "fowlkes mallows index", "informedness", "markedness"].
                           Default: ``"sensitivity"`` .
        calculation_method (bool): If true, the measurement for each sample will be calculated first.
                           If not, the confusion matrix of all samples will be accumulated first.
                           As for classification task, 'calculation_method' should be False. Default: ``False`` .
        decrease (str): The reduction method on data batch. `decrease` takes effect only when calculation_method
                        is True. Default: ``"mean"`` . Choose from:
                        ["none", "mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel"].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import ConfusionMatrixMetric
        >>>
        >>> metric = ConfusionMatrixMetric(skip_channel=True, metric_name="tpr",
        ...                                   calculation_method=False, decrease="mean")
        >>> metric.clear()
        >>> x = Tensor(np.array([[[0], [1]], [[1], [0]]]))
        >>> y = Tensor(np.array([[[0], [1]], [[0], [1]]]))
        >>> metric.update(x, y)
        >>> avg_output = metric.eval()
        >>> print(avg_output)
        [0.5]
    """
    def __init__(self,
                 skip_channel=True,
                 metric_name="sensitivity",
                 calculation_method=False,
                 decrease="mean"):
        super(ConfusionMatrixMetric, self).__init__()

        self.confusion_matrix = _ConfusionMatrix(skip_channel=skip_channel, metric_name=metric_name,
                                                 calculation_method=calculation_method, decrease=decrease)
        self.skip_channel = validator.check_value_type("skip_channel", skip_channel, [bool])
        self.calculation_method = validator.check_value_type("calculation_method", calculation_method, [bool])
        self.metric_name = validator.check_value_type("metric_name", metric_name, [str])
        decrease_list = ["none", "mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel"]
        decrease = validator.check_value_type("decrease", decrease, [str])
        self.decrease = validator.check_string(decrease, decrease_list, "decrease")
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._total_num = 0
        self._class_num = 0
        self._total_tp = 0.0
        self._total_fp = 0.0
        self._total_tn = 0.0
        self._total_fn = 0.0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Update state with predictions and targets.

        Args:
            inputs (tuple): Input `y_pred` and `y`. `y_pred` and `y` are a `Tensor`, list or numpy.ndarray.

                - y_pred (ndarray): The batch data shape is :math:`(N, C, ...)`
                  or :math:`(N, ...)`, representing onehot format
                  or category index format respectively. As for classification tasks, y_pred should have the shape [BN]
                  where N is larger than 1. As for segmentation tasks, the shape should be [BNHW] or [BNHWD].
                - y (ndarray): It must be one-hot format. The batch data shape is :math:`(N, C, ...)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError("For 'ConfusionMatrixMetric.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        if self.calculation_method:
            score, not_nans = self.confusion_matrix(y_pred, y)
            not_nans = int(not_nans.item())
            self._total_num += score.item() * not_nans
            self._class_num += not_nans
        else:
            confusion_matrix = self.confusion_matrix(y_pred, y)
            confusion_matrix, _ = _decrease_metric(confusion_matrix, "sum")
            self._total_tp += confusion_matrix[0].item()
            self._total_fp += confusion_matrix[1].item()
            self._total_tn += confusion_matrix[2].item()
            self._total_fn += confusion_matrix[3].item()

    def eval(self):
        """
        Computes confusion matrix metric.

        Returns:
            ndarray, the computed result.
        """

        if self.calculation_method:
            if self._class_num == 0:
                raise RuntimeError("The 'ConfusionMatrixMetric' can not be calculated, because the number of samples "
                                   "is 0, please check whether your inputs(predicted value, true value) are empty, or "
                                   "has called update method before calling eval method.")

            return self._total_num / self._class_num

        confusion_matrix = np.array([self._total_tp, self._total_fp, self._total_tn, self._total_fn])
        return _compute_confusion_matrix_metric(self.metric_name, confusion_matrix)


class _ConfusionMatrix:
    """
    Compute confusion matrix related metrics.

    Args:
        skip_channel (bool): Whether to skip the measurement calculation on the first channel of the predicted
                             output. Default: ``True``.
        metric_name (str): The names of indicators are in the following range. Of course, you can also set the industry
                           common aliases for these indicators.
        calculation_method (bool): If true, the measurement for each sample will be calculated first. If not, the
                                   confusion matrix for each image (the output of function '_get_confusion_matrix')
                                   will be returned. In this way, users should achieve the confusion matrixes for all
                                   images during an epochand then use '_compute_confusion_matrix_metric' to calculate
                                   the metric. Default: ``False``.
        decrease (Union[DecreaseMetric, str]): ["none", "mean", "sum", "mean_batch", "sum_batch", "mean_channel",
                                                "sum_channel"]
                                               Define the mode to reduce the calculation result of one batch of data.
                                               Decrease is used only if calculation_method is True. Default: "mean".
    """

    def __init__(self, skip_channel=True, metric_name="hit_rate", calculation_method=False,
                 decrease="mean"):
        super().__init__()
        self.skip_channel = skip_channel
        self.metric_name = metric_name
        self.calculation_method = calculation_method
        self.decrease = decrease

    def __call__(self, y_pred, y):
        """
        'y_preds' is expected to have binarized predictions and 'y' should be in one-hot format.

        Args:
            - **y_pred** (ndarray) - Input data to compute. It must be one-hot format and first dim is batch.
            - **y** (ndarray) - Ground truth to compute the metric. It must be one-hot format and first dim is batch.

        Raises:
            ValueError: If `metric_name` is empty.
            ValueError: when `y_pred` has less than two dimensions.
        """
        if not np.all(y.astype(np.uint8) == y):
            raise ValueError("For 'ConfusionMatrix.update', the true value (input[1]) must be a binarized ndarray.")

        dims = y_pred.ndim
        if dims < 2:
            raise ValueError(f"For 'ConfusionMatrix.update', the predicted value (input[0]) must have at least 2 "
                             f"dimensions, but got {dims}.")

        if dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.calculation_method:
                self.calculation_method = False

        confusion_matrix = _get_confusion_matrix(y_pred=y_pred, y=y, skip_channel=self.skip_channel)

        if self.calculation_method:
            if isinstance(self.metric_name, str):
                sub_confusion_matrix = _compute_confusion_matrix_metric(self.metric_name, confusion_matrix)
                chart, not_nans = _decrease_metric(sub_confusion_matrix, self.decrease)
                return chart, not_nans

            if not self.metric_name:
                raise ValueError("For 'ConfusionMatrix', the argument 'metric_name' cannot be None.")

            results = []
            for metric_name in self.metric_name:
                sub_confusion_matrix = _compute_confusion_matrix_metric(metric_name, confusion_matrix)
                chart, not_nans = _decrease_metric(sub_confusion_matrix, self.decrease)
                results.append(chart)
                results.append(not_nans)
            return results

        return confusion_matrix


def _get_confusion_matrix(y_pred, y, skip_channel=True):
    """
    The confusion matrix is calculated. An array of shape [BC4] is returned. The third dimension represents each channel
    of each sample in the input batch.Where B is the batch size and C is the number of classes to be calculated.

    Args:
        y_pred (ndarray): input data to compute. It must be one-hot format and first dim is batch.
                             The values should be binarized.
        y (ndarray): ground truth to compute the metric. It must be one-hot format and first dim is batch.
                    The values should be binarized.
        skip_channel (bool): whether to skip metric computation on the first channel of the predicted output.
                            Default: ``True``.

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """

    if not skip_channel:
        y = y[:, 1:] if y.shape[1] > 1 else y
        y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred

    y = y.astype(float)
    y_pred = y_pred.astype(float)
    validator.check('y_shape', y.shape, 'y_pred_shape', y_pred.shape)
    batch_size, n_class = y_pred.shape[:2]
    y_pred = y_pred.reshape(batch_size, n_class, -1)
    y = y.reshape(batch_size, n_class, -1)
    tp = ((y_pred + y) == 2).astype(float)
    tn = ((y_pred + y) == 0).astype(float)
    tp = tp.sum(axis=2)
    tn = tn.sum(axis=2)
    p = y.sum(axis=2)
    n = y.shape[-1] - p
    fn = p - tp
    fp = n - tn

    return np.stack([tp, fp, tn, fn], axis=-1)


def _decrease_mean(not_nans, chart):
    not_nans = not_nans.sum(axis=1)
    chart = np.where(not_nans > 0, chart.sum(axis=1) / not_nans, np.zeros(1, dtype=float))

    not_nans = (not_nans > 0).astype(float).sum(axis=0)
    chart = np.where(not_nans > 0, chart.sum(axis=0) / not_nans, np.zeros(1, dtype=float))

    return not_nans, chart


def _decrease_sum(not_nans, chart):
    not_nans = not_nans.sum(axis=(0, 1))
    chart = np.sum(chart, axis=(0, 1))

    return not_nans, chart


def _decrease_mean_batch(not_nans, chart):
    not_nans = not_nans.sum(axis=0)
    chart = np.where(not_nans > 0, chart.sum(axis=0) / not_nans, np.zeros(1, dtype=float))

    return not_nans, chart


def _decrease_sum_batch(not_nans, chart):
    not_nans = not_nans.sum(axis=0)
    chart = chart.sum(axis=0)

    return not_nans, chart


def _decrease_mean_channel(not_nans, chart):
    not_nans = not_nans.sum(axis=1)
    chart = np.where(not_nans > 0, chart.sum(axis=1) / not_nans, np.zeros(1, dtype=float))

    return not_nans, chart


def _decrease_sum_channel(not_nans, chart):
    not_nans = not_nans.sum(axis=1)
    chart = chart.sum(axis=1)

    return not_nans, chart


def _decrease_none(not_nans, chart):
    return not_nans, chart


def _decrease_metric(chart, decrease="mean"):
    """
    This function is used to reduce the calculated metrics for each class of each example.

    Args:
        chart (ndarray): A data table containing the calculated measurement scores for each batch and class.
                    The first two dims should be batch and class.
        decrease (str): Define the mode to reduce computation result of 1 batch data. Decrease will only be employed
                        when 'calculation_method' is True. Default: "mean".
    """

    nans = np.isnan(chart)
    not_nans = (~nans).astype(float)
    chart[nans] = 0

    decrease_dict = {"mean": _decrease_mean(not_nans, chart),
                     "sum": _decrease_sum(not_nans, chart),
                     "mean_batch": _decrease_mean_batch,
                     "sum_batch": _decrease_sum_batch(not_nans, chart),
                     "mean_channel": _decrease_mean_channel(not_nans, chart),
                     "sum_channel": _decrease_sum_channel(not_nans, chart),
                     "none": _decrease_none(not_nans, chart)}
    not_nans, chart = decrease_dict.get(decrease)

    return chart, not_nans


def _calculate_tpr(tp, p):
    """Calculate tpr."""
    return tp, p


def _calculate_tnr(tn, n):
    """Calculate tnr."""
    return tn, n


def _calculate_ppv(tp, fp):
    """Calculate ppv."""
    return tp, (tp + fp)


def _calculate_npv(tn, fn):
    """Calculate npv."""
    return tn, (tn + fn)


def _calculate_fnr(fn, p):
    """Calculate fnr."""
    return fn, p


def _calculate_fpr(fp, n):
    """Calculate fpr."""
    return fp, n


def _calculate_fdr(tp, fp):
    """Calculate fdr."""
    return fp, (fp + tp)


def _calculate_for(tn, fn):
    """Calculate for."""
    return fn, (fn + tn)


def _calculate_pt(tp, tn, p, n):
    """Calculate pt."""
    tpr = np.where(p > 0, tp / p, np.array(float("nan")))
    tnr = np.where(n > 0, tn / n, np.array(float("nan")))
    numerator = np.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
    denominator = tpr + tnr - 1.0

    return numerator, denominator


def _calculate_ts(tp, fp, fn):
    """Calculate ts."""
    return tp, (tp + fn + fp)


def _calculate_acc(tp, tn, p, n):
    """Calculate acc."""
    return (tp + tn), (p + n)


def _calculate_ba(tp, tn, p, n):
    """Calculate ba."""
    tpr = np.where(p > 0, tp / p, np.array(float("nan")))
    tnr = np.where(n > 0, tn / n, np.array(float("nan")))
    numerator, denominator = (tpr + tnr), 2.0

    return numerator, denominator


def _calculate_f1(tp, fp, fn):
    """Calculate f1."""
    return tp * 2.0, (tp * 2.0 + fn + fp)


def _calculate_mcc(tp, fp, tn, fn):
    """Calculate mcc."""
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator, denominator


def _calculate_fm(tp, fp, p):
    """Calculate fm."""
    tpr = np.where(p > 0, tp / p, np.array(float("nan")))
    ppv = np.where((tp + fp) > 0, tp / (tp + fp), np.array(float("nan")))
    numerator = np.sqrt(ppv * tpr)
    denominator = 1.0

    return numerator, denominator


def _calculate_bm(tp, tn, p, n):
    """Calculate bm."""
    tpr = np.where(p > 0, tp / p, np.array(float("nan")))
    tnr = np.where(n > 0, tn / n, np.array(float("nan")))
    numerator = tpr + tnr - 1.0
    denominator = 1.0

    return numerator, denominator


def _calculate_mk(tp, fp, tn, fn):
    """Calculate mk."""
    ppv = np.where((tp + fp) > 0, tp / (tp + fp), np.array(float("nan")))
    npv = np.where((tn + fn) > 0, tn / (tn + fn), np.array(float("nan")))
    npv = tn / (tn + fn)
    numerator = ppv + npv - 1.0
    denominator = 1.0

    return numerator, denominator


def _compute_confusion_matrix_metric(metric_name, confusion_matrix):
    """
    This function is used to compute confusion matrix related metric.

    Args:
        metric_name (str): Refer to conflusionmatrixmetric 'metric_name'. Some of the metrics have multiple aliases
                           (as shown in the wikipedia page aforementioned), and you can also input those names instead.
        confusion_matrix (ndarray): Refer to '_get_confusion_matrix'.

    Raises:
        ValueError: when the size of the last dimension of confusion_matrix is not 4.
        NotImplementedError: when specify a not implemented metric_name.

    """

    metric = _check_metric_name(metric_name)

    input_dim = confusion_matrix.ndim
    if input_dim == 1:
        confusion_matrix = np.expand_dims(confusion_matrix, 0)
    if confusion_matrix.shape[-1] != 4:
        raise ValueError(f"For 'ConfusionMatrix', the size of the last dimension of confusion_matrix must be 4, "
                         f"but got {confusion_matrix.shape[-1]}.")

    tp = confusion_matrix[..., 0]
    fp = confusion_matrix[..., 1]
    tn = confusion_matrix[..., 2]
    fn = confusion_matrix[..., 3]
    p = tp + fn
    n = fp + tn

    metric_name_dict = {"tpr": _calculate_tpr(tp, p),
                        "tnr": _calculate_tnr(tn, n),
                        "ppv": _calculate_ppv(tp, fp),
                        "npv": _calculate_npv(tn, fn),
                        "fnr": _calculate_fnr(fn, p),
                        "fpr": _calculate_fpr(fp, n),
                        "fdr": _calculate_fdr(tp, fp),
                        "for": _calculate_for(tn, fn),
                        "pt": _calculate_pt(tp, tn, p, n),
                        "ts": _calculate_ts(tp, fp, fn),
                        "acc": _calculate_acc(tp, tn, p, n),
                        "ba": _calculate_ba(tp, tn, p, n),
                        "f1": _calculate_f1(tp, fp, fn),
                        "mcc": _calculate_mcc(tp, fp, tn, fn),
                        "fm": _calculate_fm(tp, fp, p),
                        "bm": _calculate_bm(tp, tn, p, n),
                        "mk": _calculate_mk(tp, fp, tn, fn)}
    numerator, denominator = metric_name_dict.get(metric)

    if isinstance(denominator, np.ndarray):
        result = np.where(denominator != 0, numerator / denominator, np.array(float("nan")))
    else:
        result = numerator / denominator
    return result


def _check_metric_name(metric_name):
    """
    There are many metrics related to confusion matrix, and some of the metrics have more than one names. In addition,
    some of the names are very long. Therefore, this function is used to check and simplify the name.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    metric_name_dict = {"sensitivity": "tpr",
                        "recall": "tpr",
                        "hit_rate": "tpr",
                        "true_positive_rate": "tpr",
                        "tpr": "tpr",
                        "specificity": "tnr",
                        "selectivity": "tnr",
                        "true_negative_rate": "tnr",
                        "tnr": "tnr",
                        "precision": "ppv",
                        "positive_predictive_value": "ppv",
                        "ppv": "ppv",
                        "negative_predictive_value": "npv",
                        "npv": "npv",
                        "miss_rate": "fnr",
                        "false_negative_rate": "fnr",
                        "fnr": "fnr",
                        "fall_out": "fpr",
                        "false_positive_rate": "fpr",
                        "fpr": "fpr",
                        "false_discovery_rate": "fdr",
                        "fdr": "fdr",
                        "false_omission_rate": "for",
                        "for": "for",
                        "prevalence_threshold": "pt",
                        "pt": "pt",
                        "threat_score": "ts",
                        "critical_success_index": "ts",
                        "ts": "ts",
                        "csi": "ts",
                        "accuracy": "acc",
                        "acc": "acc",
                        "balanced_accuracy": "ba",
                        "ba": "ba",
                        "f1_score": "f1",
                        "f1": "f1",
                        "matthews_correlation_coefficient": "mcc",
                        "mcc": "mcc",
                        "fowlkes_mallows_index": "fm",
                        "fm": "fm",
                        "informedness": "bm",
                        "bookmaker_informedness": "bm",
                        "bm": "bm",
                        "markedness": "mk",
                        "deltap": "mk",
                        "mk": "mk"}

    metric_name_info = metric_name_dict.get(metric_name)

    if metric_name_info is None:
        raise NotImplementedError("The metric is not implemented.")

    return metric_name_info
