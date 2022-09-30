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
"""Callback related classes and functions."""
from __future__ import absolute_import

from mindspore.train.callback._callback import Callback
from mindspore.train.callback._callback import CallbackManager as _CallbackManager
from mindspore.train.callback._callback import InternalCallbackParam as _InternalCallbackParam
from mindspore.train.callback._callback import RunContext
from mindspore.train.callback._callback import checkpoint_cb_for_save_op as _checkpoint_cb_for_save_op
from mindspore.train.callback._callback import set_cur_net as _set_cur_net
from mindspore.train.callback._checkpoint import CheckpointConfig
from mindspore.train.callback._checkpoint import CheckpointManager as _CheckpointManager
from mindspore.train.callback._checkpoint import ModelCheckpoint
from mindspore.train.callback._loss_monitor import LossMonitor
from mindspore.train.callback._time_monitor import TimeMonitor
from mindspore.train.callback._summary_collector import SummaryCollector
from mindspore.train.callback._lr_scheduler_callback import LearningRateScheduler
from mindspore.train.callback._landscape import SummaryLandscape
from mindspore.train.callback._history import History
from mindspore.train.callback._lambda_callback import LambdaCallback
from mindspore.train.callback._early_stop import EarlyStopping
from mindspore.train.callback._reduce_lr_on_plateau import ReduceLROnPlateau
from mindspore.train.callback._on_request_exit import OnRequestExit
from mindspore.train.callback._backup_and_restore import BackupAndRestore

__all__ = ["Callback", "LossMonitor", "TimeMonitor", "ModelCheckpoint",
           "SummaryCollector", "CheckpointConfig", "RunContext", "LearningRateScheduler", "SummaryLandscape",
           "History", "LambdaCallback", "ReduceLROnPlateau", "EarlyStopping", "OnRequestExit", "BackupAndRestore"]
