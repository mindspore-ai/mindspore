# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Utils module."""
from __future__ import absolute_import
from mindspore._c_expression import _reuse_data_ptr
from .stress_detect import stress_detect
from .utils import ExitByRequest, RSCPluginHandle, TFTCommValue, _tft_handler
from .runtime_execution_order_check import runtime_execution_order_check, comm_exec_order_check
from .sdc_detect import sdc_detect_start, sdc_detect_stop, get_sdc_detect_result
from . import dryrun
from .dlpack import from_dlpack, to_dlpack

# Symbols from utils module.
__all__ = ["stress_detect", "ExitByRequest", "runtime_execution_order_check", "dryrun", "_reuse_data_ptr",
           "_tft_handler", "comm_exec_order_check", "sdc_detect_start", "sdc_detect_stop", "get_sdc_detect_result",
           "RSCPluginHandle", "TFTCommValue", "from_dlpack", "to_dlpack"]
