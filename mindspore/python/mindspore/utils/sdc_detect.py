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
"""SDC detect."""
import mindspore.tools
from mindspore.common._decorator import deprecated


@deprecated("2.7.1", "mindspore.tools.sdc_detect_start", module_prefix="mindspore.utils.")
def sdc_detect_start():
    """
    This api will be deprecated and removed in future versions, please use the api
    :func:`mindspore.tools.sdc_detect_start` instead.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.utils import sdc_detect_start
        >>> sdc_detect_start()
    """
    return mindspore.tools.sdc_detect_start()


@deprecated("2.7.1", "mindspore.tools.sdc_detect_stop", module_prefix="mindspore.utils.")
def sdc_detect_stop():
    """
    This api will be deprecated and removed in future versions, please use the api
    :func:`mindspore.tools.sdc_detect_stop` instead.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.utils import sdc_detect_stop
        >>> sdc_detect_stop()
    """
    return mindspore.tools.sdc_detect_stop()


@deprecated("2.7.1", "mindspore.tools.get_sdc_detect_result", module_prefix="mindspore.utils.")
def get_sdc_detect_result():
    """
    This api will be deprecated and removed in future versions, please use the api
    :func:`mindspore.tools.get_sdc_detect_result` instead.

    Returns:
        bool, indicating whether silent data corruption has occurred after detection start.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.utils import get_sdc_detect_result
        >>> result = get_sdc_detect_result()
        >>> print(result)
        False
    """
    return mindspore.tools.get_sdc_detect_result()
