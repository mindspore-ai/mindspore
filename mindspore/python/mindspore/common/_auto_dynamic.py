# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""convert inputs to dynamic shape automatically at the first round. Method in this file is only used for test."""
import os
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor


def is_auto_dynamic_shape():
    """this is used only for test"""
    return os.getenv("MS_DEV_AUTO_DYNAMIC_SHAPE") == "on"


def is_auto_dynamic_rank():
    """this is used only for test"""
    return os.getenv("MS_DEV_AUTO_DYNAMIC_RANK") == "on"


def is_auto_dynamic():
    """this is used only for test"""
    return is_auto_dynamic_shape() or is_auto_dynamic_rank()


def convert_inputs_to_dynamic(*inputs):
    """this is used only for test"""
    dyn_inputs = list(inputs)
    if not dyn_inputs:
        return None
    for idx, net_input in enumerate(inputs):
        if isinstance(net_input, Tensor) and not isinstance(net_input, Parameter):
            shp = net_input.shape
            if not shp:
                dyn_inputs[idx] = net_input
                continue
            if is_auto_dynamic_rank():
                dyn_tensor = Tensor(shape=None, dtype=net_input.dtype)
            else:
                dyn_shape = [None for _ in net_input.shape]
                dyn_tensor = Tensor(shape=dyn_shape, dtype=net_input.dtype)
            dyn_inputs[idx] = dyn_tensor

    return tuple(dyn_inputs)


def convert_new_shapes(dataset_shapes):
    """this is used only for test"""
    new_shapes = []
    for shape in dataset_shapes:
        if is_auto_dynamic_rank():
            new_shape = [-2]
        else:
            new_shape = [-1 for _ in shape]
        new_shapes.append(new_shape)
    return new_shapes
