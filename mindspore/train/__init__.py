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
High-Level training interfaces.

Helper functions in train piplines.
"""
from .model import Model
from .dataset_helper import DatasetHelper, connect_network_with_dataset
from . import amp
from .amp import build_train_network
from .loss_scale_manager import LossScaleManager, FixedLossScaleManager, DynamicLossScaleManager
from .serialization import save_checkpoint, load_checkpoint, load_param_into_net, export, load, parse_print,\
    build_searched_strategy, merge_sliced_parameter, load_distributed_checkpoint

__all__ = ["Model", "DatasetHelper", "amp", "connect_network_with_dataset", "build_train_network", "LossScaleManager",
           "FixedLossScaleManager", "DynamicLossScaleManager", "save_checkpoint", "load_checkpoint",
           "load_param_into_net", "export", "load", "parse_print", "build_searched_strategy", "merge_sliced_parameter",
           "load_distributed_checkpoint"]
