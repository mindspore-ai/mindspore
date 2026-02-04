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
"""Runtime stream class"""
from mindspore._c_expression import get_device_limit as get_device_limit_
from mindspore._c_expression import set_device_limit as set_device_limit_


def get_device_limit(device):
    r"""
    Return selected device limit core num.

    Note:
        - This interface will synchronize the operator issuance, which may affect performance.
        - Only support PyNative mode, Graph mode is not currently supported.

    Args:
        device (int): selected device id.

    Returns:
        limit info (dict), device limit core num.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> ms.runtime.get_device_limit(0)
    """
    cube_num, vector_num = get_device_limit_(device)
    return {"cube_core_num": cube_num, "vector_core_num": vector_num}


def set_device_limit(device, cube_num=-1, vector_num=-1):
    r"""
    Sets selected device limit.

    Note:
        - This interface will synchronize the operator issuance, which may affect performance.
        - Only support PyNative mode, Graph mode is not currently supported.

    Args:
        device (int): selected set device id.
        cube_num (int, optional): set cube num for device. Default is ``-1``, indicating that it is not set.
        vector_num (int, optional): set vector num for device. Default is ``-1``, indicating that it is not set.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> ms.runtime.set_device_limit(0, 8, 8)
    """
    set_device_limit_(device, cube_num, vector_num)
