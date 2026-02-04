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
import mindspore.tools
from mindspore.common._decorator import deprecated


@deprecated("2.7.1", "mindspore.tools.stress_detect", module_prefix="mindspore.utils.")
def stress_detect(detect_type="aic"):
    """
    This api will be deprecated and removed in future versions, please use the api
    :func:`mindspore.tools.stress_detect` instead.

    Args:
        detect_type (str, optional): The type of stress test to perform. There are two options available: ``'aic'`` and
            ``'hccs'``, which perform AiCore and HCCS link stress tests on the device, respectively. Default: "aic".

    Returns:
        int, the return value represents the error type. 0 indicates normal. 1 indicates failure to start some or
        all test cases. 2 indicates a hardware failure, and it is recommended to replace the device.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.utils import stress_detect
        >>> ret = stress_detect()
        >>> print(ret)
        0
    """
    return mindspore.tools.stress_detect(detect_type)
