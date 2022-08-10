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
Wrap cells for networks.

Use the Wrapper to combine the loss or build the training steps.
"""
from __future__ import absolute_import

from mindspore.nn.wrap.cell_wrapper import ForwardValueAndGrad, TrainOneStepCell, WithLossCell, WithGradCell, \
    WithEvalCell, ParameterUpdate, GetNextSingleOp, VirtualDatasetCellTriple, MicroBatchInterleaved, PipelineCell
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell,\
    DynamicLossScaleUpdateCell, FixedLossScaleUpdateCell
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn.layer.timedistributed import TimeDistributed


__all__ = [
    "TimeDistributed",
    "ForwardValueAndGrad",
    "TrainOneStepCell",
    "WithLossCell",
    "WithGradCell",
    "MicroBatchInterleaved",
    "PipelineCell",
    "WithEvalCell",
    "GetNextSingleOp",
    "TrainOneStepWithLossScaleCell",
    "DistributedGradReducer",
    "ParameterUpdate",
    "DynamicLossScaleUpdateCell",
    "FixedLossScaleUpdateCell",
    "VirtualDatasetCellTriple"
    ]
