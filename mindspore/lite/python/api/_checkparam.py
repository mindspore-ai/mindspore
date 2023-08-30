# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Check parameters.
"""


def check_isinstance(arg_name, arg_value, classes, enable_none=False):
    """Check arg isinstance of classes"""
    if enable_none:
        if arg_value is None:
            return arg_value
    if not isinstance(arg_value, classes):
        raise TypeError(f"{arg_name} must be {classes.__name__}, but got {format(type(arg_value))}.")
    return arg_value


def check_list_of_element(arg_name, arg_value, ele_classes, enable_none=False):
    """Check arg isinstance of classes"""
    if enable_none:
        if arg_value is None:
            return arg_value
    if not isinstance(arg_value, list):
        raise TypeError(f"{arg_name} must be list, but got {format(type(arg_value))}.")
    for i, element in enumerate(arg_value):
        if not isinstance(element, ele_classes):
            raise TypeError(f"{arg_name} element must be {ele_classes.__name__}, but got "
                            f"{type(element)} at index {i}.")
    return arg_value


def check_input_shape(input_shape_name, input_shape, enable_none=False):
    """Check input_shape's type is dict{str: list[int]}"""
    if enable_none:
        if input_shape is None:
            return input_shape
    if not isinstance(input_shape, dict):
        raise TypeError(f"{input_shape_name} must be dict, but got {format(type(input_shape))}.")
    for key in input_shape:
        if not isinstance(key, str):
            raise TypeError(f"{input_shape_name} key must be str, but got {format(type(input_shape))}.")
        if not isinstance(input_shape[key], list):
            raise TypeError(f"{input_shape_name} value must be list, but got "
                            f"{type(input_shape[key])} at key {key}.")
        for j, element in enumerate(input_shape[key]):
            if not isinstance(element, int):
                raise TypeError(f"{input_shape_name} value's element must be int, but got "
                                f"{type(element)} at index {j}.")
    return input_shape


def check_config_info(config_info_name, config_info, enable_none=False):
    """Check config_info's type is dict{str: str}"""
    if enable_none:
        if config_info is None:
            return config_info
    if not isinstance(config_info, dict):
        raise TypeError(f"{config_info_name} must be dict, but got {format(type(config_info))}.")
    for key in config_info:
        if not isinstance(key, str):
            raise TypeError(f"{config_info_name} key must be str, but got {type(key)} at key {key}.")
        if not isinstance(config_info[key], str):
            raise TypeError(f"{config_info_name} val must be str, but got "
                            f"{type(config_info[key])} at key {key}.")
    return config_info


def check_tensor_input_param(shape=None, device=None):
    """Check tensor input param"""
    if shape is not None:
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape must be list or tuple, but got {type(shape)}.")
        for i, element in enumerate(shape):
            if not isinstance(element, int):
                raise TypeError(f"shape element must be int, but got {type(element)} at index {i}.")
    if device is None:
        return
    if device is not None and not isinstance(device, str):
        raise TypeError(f"device must be str, but got {type(device)}.")
    split_device = device.split(":")
    if len(split_device) > 2:
        raise TypeError(f"device must be 'ascend:index', eg: 'ascend:0'")
    if len(split_device) > 0 and split_device[0] != "ascend":
        raise TypeError(f"now only support ascend device.")
    if len(split_device) == 2 and not split_device[1].isdigit():
        raise TypeError(f"device id should >= 0.")
