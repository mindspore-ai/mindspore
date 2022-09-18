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
Metrics from mindspore.train.metrics
"""

from mindspore.train.metrics import Accuracy, HausdorffDistance, MAE, MSE, Metric, \
    rearrange_inputs, Precision, Recall, Fbeta, F1, Dice, ROC, auc, \
    TopKCategoricalAccuracy, Top1CategoricalAccuracy, Top5CategoricalAccuracy, Loss, \
    MeanSurfaceDistance, RootMeanSquareDistance, BleuScore, CosineSimilarity, \
    OcclusionSensitivity, Perplexity, ConfusionMatrixMetric, ConfusionMatrix, \
    names, get_metric_fn, get_metrics

__all__ = [
    "names",
    "get_metric_fn",
    "get_metrics",
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
