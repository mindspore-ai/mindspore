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
"""tbe process"""
import sys
import os
from te.platform.cce_conf import te_set_version
from .tbe_common import get_args, get_built_in_impl_path, TBEException

build_in_impl_path = get_built_in_impl_path()


def _op_select_format(kernel_info):
    """
    call op's op_select_format to get op supported format

    Args:
        kernel_info (dict): kernel info load by json string

    Returns:
        op supported format
    """
    try:
        op_name = kernel_info['op_info']['name']
        te_set_version(kernel_info["op_info"]["socVersion"])
        impl_path = build_in_impl_path
        custom_flag = False
        if 'impl_path' in kernel_info and kernel_info['impl_path'] is not None:
            op_impl_path = os.path.realpath(kernel_info['impl_path'])
            if os.path.isfile(op_impl_path):
                path, file_name = os.path.split(op_impl_path)
                op_name, _ = os.path.splitext(file_name)
                impl_path = path
                custom_flag = True
        if impl_path not in sys.path:
            sys.path.insert(0, impl_path)

        if custom_flag:
            op_module = __import__(op_name)
        else:
            op_module = __import__("impl." + op_name, globals(), locals(), [op_name], 0)
        # get function
        if not hasattr(op_module, "op_select_format"):
            return ""
        op_func = getattr(op_module, "op_select_format", None)

        # call function
        inputs_args = get_args(kernel_info['op_info'], 'inputs')
        outputs_args = get_args(kernel_info['op_info'], 'outputs')
        attrs_args = get_args(kernel_info['op_info'], 'attrs')
        kernel_name = kernel_info['op_info']['kernel_name']
        ret = op_func(*inputs_args, *outputs_args, *attrs_args, kernel_name=kernel_name)

    except Exception as e:
        raise TBEException(str(e))

    return ret


def _check_supported(kernel_info):
    """
    call op's check_supported to check supported or not

    Args:
        kernel_info (dict): kernel info load by json string

    Returns:
        bool: check result, true or false
    """
    try:
        op_name = kernel_info['op_info']['name']
        is_dynamic_shape = kernel_info['op_info']['is_dynamic_shape']
        te_set_version(kernel_info["op_info"]["socVersion"])
        impl_path = build_in_impl_path
        custom_flag = False
        if 'impl_path' in kernel_info and kernel_info['impl_path'] is not None:
            op_impl_path = os.path.realpath(kernel_info['impl_path'])
            if os.path.isfile(op_impl_path):
                path, file_name = os.path.split(op_impl_path)
                op_name, _ = os.path.splitext(file_name)
                impl_path = path
                custom_flag = True
        if impl_path not in sys.path:
            sys.path.insert(0, impl_path)

        if custom_flag:
            op_module = __import__(op_name)
        elif is_dynamic_shape:
            op_module = __import__("impl.dynamic." + op_name, globals(), locals(), [op_name], 0)
        else:
            op_module = __import__("impl." + op_name, globals(), locals(), [op_name], 0)

        # get function
        if not hasattr(op_module, "check_supported"):
            return ""
        op_func = getattr(op_module, "check_supported", None)

        # call function
        inputs_args = get_args(kernel_info['op_info'], 'inputs')
        outputs_args = get_args(kernel_info['op_info'], 'outputs')
        attrs_args = get_args(kernel_info['op_info'], 'attrs')
        kernel_name = kernel_info['op_info']['kernel_name']
        ret = op_func(*inputs_args, *outputs_args, *attrs_args, kernel_name=kernel_name)

    except Exception as e:
        raise TBEException(str(e))

    return ret
