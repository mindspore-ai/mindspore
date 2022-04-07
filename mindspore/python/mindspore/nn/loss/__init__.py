# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Loss.

Cells of loss function. Loss function in machine learning is the target of the model.
It shows how well the model works on a dataset and the optimization target which the optimizer is searching.
"""
from __future__ import absolute_import

from mindspore.nn.loss.loss import LossBase, L1Loss, CTCLoss, MSELoss, SmoothL1Loss, SoftMarginLoss, FocalLoss, \
    SoftmaxCrossEntropyWithLogits, BCELoss, MultiMarginLoss, CosineEmbeddingLoss, \
    SampledSoftmaxLoss, TripletMarginWithDistanceLoss,\
    PoissonNLLLoss, MultiLabelSoftMarginLoss, DiceLoss, BCEWithLogitsLoss, MultiClassDiceLoss, \
    RMSELoss, MAELoss, HuberLoss, CrossEntropyLoss, NLLLoss, KLDivLoss, MarginRankingLoss, GaussianNLLLoss, \
    HingeEmbeddingLoss, MultilabelMarginLoss, TripletMarginLoss


__all__ = ['LossBase', 'L1Loss', 'CTCLoss', 'MSELoss', 'SmoothL1Loss', 'SoftMarginLoss', 'FocalLoss',
           'SoftmaxCrossEntropyWithLogits', 'BCELoss', 'BCEWithLogitsLoss', 'MultiMarginLoss',
           'CosineEmbeddingLoss', 'SampledSoftmaxLoss', 'TripletMarginWithDistanceLoss', 'PoissonNLLLoss',
           'MultiLabelSoftMarginLoss', 'DiceLoss', 'MultiClassDiceLoss', 'MultilabelMarginLoss',
           'RMSELoss', 'MAELoss', 'HuberLoss', 'CrossEntropyLoss', 'NLLLoss', 'KLDivLoss', 'MarginRankingLoss',
           'GaussianNLLLoss', 'HingeEmbeddingLoss', 'TripletMarginLoss']
