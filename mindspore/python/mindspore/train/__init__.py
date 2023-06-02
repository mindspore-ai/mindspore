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

Helper functions in train pipelines.
"""
from __future__ import absolute_import

from mindspore.train.model import Model
from mindspore.train.dataset_helper import DatasetHelper, connect_network_with_dataset
from mindspore.train import amp
from mindspore.train.amp import build_train_network
from mindspore.train.loss_scale_manager import LossScaleManager, FixedLossScaleManager, DynamicLossScaleManager
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net, export, \
    load, parse_print, build_searched_strategy, merge_sliced_parameter, load_distributed_checkpoint, \
    async_ckpt_thread_status, restore_group_info_list, convert_model, obfuscate_model, export_split_mindir
from mindspore.train.callback import Callback, LossMonitor, TimeMonitor, ModelCheckpoint, SummaryCollector, \
    CheckpointConfig, RunContext, LearningRateScheduler, SummaryLandscape, \
    History, LambdaCallback, ReduceLROnPlateau, EarlyStopping, OnRequestExit, BackupAndRestore
from mindspore.train.summary import SummaryRecord
from mindspore.train.train_thor import ConvertNetUtils, ConvertModelUtils
from mindspore.train.metrics import *
from mindspore.train.data_sink import data_sink

__all__ = ["Model", "DatasetHelper", "connect_network_with_dataset", "build_train_network", "LossScaleManager",
           "FixedLossScaleManager", "DynamicLossScaleManager", "save_checkpoint", "load_checkpoint",
           "load_param_into_net", "export", "load", "export_split_mindir", "parse_print", "build_searched_strategy",
           "merge_sliced_parameter", "load_distributed_checkpoint", "async_ckpt_thread_status",
           "restore_group_info_list", "convert_model", "data_sink", "obfuscate_model"]
__all__.extend(callback.__all__)
__all__.extend(summary.__all__)
__all__.extend(train_thor.__all__)
__all__.extend(metrics.__all__)
