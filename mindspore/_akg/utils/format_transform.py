# Copyright 2019 Huawei Technologies Co., Ltd
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

"""format transform function"""
import _akg

def refine_reduce_axis(input, axis):
    """make reduce axis legal."""
    shape = get_shape(input)
    if axis is None:
        axis = [i for i in range(len(shape))]
    elif isinstance(axis, int):
        axis = [axis]
    elif not isinstance(axis, (tuple, list)):
        raise TypeError("axis must be one of the type int,tuple,list or None")

    if len(axis) > len(shape):
        raise ValueError("axis size must not larger than shape size")

    axis = list(axis)

    for i, _ in enumerate(axis):
        if axis[i] < 0:
            axis[i] += len(shape)

        if axis[i] >= len(shape):
            raise ValueError(("axis value-{} exceeds len(axis) which is invalid".format(axis[i])))

    axis.sort(reverse=True)

    return axis


def get_shape_from_tensor(data):
    """translate _akg.tvm.shape to list type in python."""
    tvm_shape = data.shape
    py_shape = []
    for i in tvm_shape:
        if isinstance(i, _akg.tvm.expr.Var):
            py_shape.append(i)
        else:
            py_shape.append(i.value)
    return py_shape


def tvm_shape_to_list(tvm_shape):
    """translate _akg.tvm.shape to list type in python."""
    py_shape = []
    for i in tvm_shape:
        if isinstance(i, _akg.tvm.expr.Var):
            py_shape.append(i)
        else:
            py_shape.append(i.value)
    return py_shape


def get_shape(data):
    """get shape and save it as list."""
    if isinstance(data, _akg.tvm.tensor.Tensor):
        shape = get_shape_from_tensor(data)
    elif isinstance(data, _akg.tvm.container.Array):
        shape = tvm_shape_to_list(data)
    elif isinstance(data, int):
        shape = [data]
    elif isinstance(data, (tuple, list)):
        shape = list(data)
    else:
        raise TypeError("Refine axis does not support type {} for now.".format(type(data)))
    return shape
