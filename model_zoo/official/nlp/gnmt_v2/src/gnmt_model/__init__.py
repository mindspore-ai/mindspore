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
"""GNMTv2 Init."""
from config.config import GNMTConfig
from .gnmt import GNMT
from .attention import BahdanauAttention
from .gnmt_for_train import GNMTTraining, LabelSmoothedCrossEntropyCriterion, \
    GNMTNetworkWithLoss, GNMTTrainOneStepWithLossScaleCell
from .gnmt_for_infer import infer
from .bleu_calculate import bleu_calculate

__all__ = [
    "infer",
    "GNMTTraining",
    "LabelSmoothedCrossEntropyCriterion",
    "GNMTTrainOneStepWithLossScaleCell",
    "GNMTNetworkWithLoss",
    "GNMT",
    "BahdanauAttention",
    "GNMTConfig",
    "bleu_calculate"
]
