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
from .dataset import load_dataset
from .dataset import bi_data_loader
from .gnmt_model import GNMT, infer, GNMTNetworkWithLoss, GNMTTrainOneStepWithLossScaleCell
from .gnmt_model import LabelSmoothedCrossEntropyCriterion

__all__ = [
    "load_dataset",
    "bi_data_loader",
    "GNMT",
    "infer",
    "GNMTNetworkWithLoss",
    "GNMTTrainOneStepWithLossScaleCell",
    "LabelSmoothedCrossEntropyCriterion"
]
