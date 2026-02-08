# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Stress detect."""
from mindspore import _c_expression
from mindspore import log as logger
from mindspore.communication import init, create_group, get_rank
from mindspore.communication import get_local_rank_size


def stress_detect(detect_type="aic"):
    """
    Used to detect whether there are faults in hardware accuracy or communication between links.
    The common usage scenario is to initiate a new thread or call this interface through a Callback function
    at each step or when saving checkpoints, to check whether hardware malfunctions could affect accuracy.

    Args:
        detect_type (str, optional): The type of stress test to perform. There are two options available: ``'aic'`` and
            ``'hccs'``, which perform AiCore and HCCS link stress tests on the device, respectively. Default: "aic".

    Returns:
        int, the return value represents the error type. 0 indicates normal. 1 indicates failure to start some or
        all test cases. 2 indicates a hardware failure, and it is recommended to replace the device.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.tools import stress_detect
        >>> ret = stress_detect()
        >>> print(ret)
        0
    """
    if detect_type not in ["aic", "hccs"]:
        logger.error(f"For stress detect, detection type must be 'aic' or 'hccs'."
                     f"But got {detect_type}. Exiting stress detect.")
        return 1

    if detect_type == "aic":
        return _c_expression.stress_detect("aic")

    init()
    local_ranks = []
    local_rank_size = get_local_rank_size()
    node_num = get_rank() // local_rank_size
    for i in range(local_rank_size):
        local_ranks.append(local_rank_size * node_num + i)
    if get_rank() in local_ranks:
        group = f"new_group_{node_num}"
        create_group(group, local_ranks)

    return _c_expression.stress_detect(group)
