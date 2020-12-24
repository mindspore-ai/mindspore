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

from ._callback import Callback
from ._callback import CallbackManager as _CallbackManager
from ._callback import InternalCallbackParam as _InternalCallbackParam
from ._callback import RunContext
from ._callback import checkpoint_cb_for_save_op as _checkpoint_cb_for_save_op
from ._callback import set_cur_net as _set_cur_net
from ._checkpoint import CheckpointConfig
from ._checkpoint import CheckpointManager as _CheckpointManager
from ._checkpoint import ModelCheckpoint
from ._loss_monitor import LossMonitor
from ._time_monitor import TimeMonitor
from ._summary_collector import SummaryCollector
from ._lr_scheduler_callback import LearningRateScheduler

__all__ = ["Callback", "LossMonitor", "TimeMonitor", "ModelCheckpoint",
           "SummaryCollector", "CheckpointConfig", "RunContext", "LearningRateScheduler"]
