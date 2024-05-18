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

"""Hardware memory interfaces."""
from mindspore._c_expression import _memory_stats
from mindspore import log as logger
from .device import _check_inputs_validation, is_initialized


@_check_inputs_validation
def memory_stats(device_target=None):
    """
    Get memory pool's statistics.
    """
    if not is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return empty dict.")
        return {}
    return _memory_stats(device_target)
