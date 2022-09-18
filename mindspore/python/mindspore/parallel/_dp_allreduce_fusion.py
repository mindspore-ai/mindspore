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
"""Data parallel allreduce fusion"""
from __future__ import absolute_import
from __future__ import division

import ctypes

_MAX_GROUP_NAME_LEN = 127
_HCCL_LIB = 'libhccl.so'


def _load_lib():
    try:
        hccl_lib = ctypes.CDLL(_HCCL_LIB)
    except Exception:
        raise RuntimeError('Get hccl lib error')

    return hccl_lib


def _c_str(string):
    """Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))


def _c_array(ctype, values):
    """Create ctypes array from a python array."""
    return (ctype * len(values))(*values)


def _set_fusion_strategy_by_idx(idx_list, group="hccl_world_group"):
    """
    A function set gradient segment strategy according to the index list.

    Note:
        In the back propagation,
        the fusion of the allreduce operators with a fusion attribute equals 1,
        will be performed according to the idx_list,
        to achieve the effect of parallel between calculation and communication.

    Args:
        idx_list (list): The index list of the gradient.
        group (str): The hccl communication group.

    Raises:
        TypeError: If group is not a python str.
        TypeError: If idx_list is not a python list.
        TypeError: If type of idx_list item is not int.
        ValueError: If group name length is out of range.
        ValueError: If idx_list length is 0.
        ValueError: If idx_list item is less than 0.
        RuntimeError: If allreduce split failed.
    """
    try:
        lib_ctype = _load_lib()
    except RuntimeError:
        import hccl_test.manage.api as hccl
        hccl.set_fusion_strategy_by_idx()
        return
    finally:
        pass
    if isinstance(group, str):
        group_len = len(group)
        if (group_len > _MAX_GROUP_NAME_LEN or group_len == 0):
            raise ValueError('Group name len is out of range {_MAX_GROUP_NAME_LEN}')
    else:
        raise TypeError('Group must be a python str')

    if isinstance(idx_list, list):
        idx_len = len(idx_list)
        if idx_len == 0:
            raise ValueError('idx_list length is 0')
    else:
        raise TypeError('idx_list must be a python list')

    for idx in idx_list:
        if isinstance(idx, int):
            if idx < 0:
                raise ValueError('Idx < 0')
        else:
            raise TypeError('Idx in idx_list is invalid')

    c_array_idx_list = _c_array(ctypes.c_uint, idx_list)
    c_idx_num = ctypes.c_uint(len(idx_list))
    c_group = _c_str(group)
    ret = lib_ctype.HcomSetGradFusionByIndex(c_group, c_idx_num, c_array_idx_list)
    if ret != 0:
        raise RuntimeError('Allreduce split error')


def _set_fusion_strategy_by_size(data_size_list, group="hccl_world_group"):
    """
    A function set gradient segment strategy according to the data size percentage list.

    Note:
        In the back propagation,
        the fusion of the allreduce operators with a fusion attribute equals 1,
        will be performed according to data_size_list,
        to achieve the effect of parallel between calculation and communication.

    Args:
        data_size_list (list): The data size percentage list of the gradient.
        group (str): The hccl communication group.

    Raises:
        TypeError: If group is not a python str.
        TypeError: If data_size_list is not a python list.
        TypeError: If type of data_size_list item is not int or float.
        ValueError: If group name length is out of range.
        ValueError: If data_size_list length is 0.
        ValueError: If data_size_list item is less than 0.
        RuntimeError: If allreduce split failed.
    """
    try:
        lib_ctype = _load_lib()
    except RuntimeError:
        import hccl_test.manage.api as hccl
        hccl.set_fusion_strategy_by_size()
        return
    finally:
        pass

    if isinstance(group, str):
        group_len = len(group)
        if group_len > _MAX_GROUP_NAME_LEN or group_len == 0:
            raise ValueError('Group name is out of range {_MAX_GROUP_NAME_LEN}')
    else:
        raise TypeError('Group must be a python str')
    if isinstance(data_size_list, list):
        len_data_size = len(data_size_list)
        if len_data_size == 0:
            raise ValueError('data_size_list length is 0')
    else:
        raise TypeError('data_size_list must be a python list')
    for data_size in data_size_list:
        if not isinstance(data_size, (int, float)):
            raise TypeError('data_size in data_size_list is invalid')

    c_array_size_list = _c_array(ctypes.c_float, data_size_list)
    c_size_num = ctypes.c_uint(len(data_size_list))
    c_group = _c_str(group)
    ret = lib_ctype.hcom_set_split_strategy_by_size(c_group, c_size_num, c_array_size_list)
    if ret != 0:
        raise RuntimeError('Allreduce split error')
