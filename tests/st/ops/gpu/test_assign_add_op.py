# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


class AssignAdd(nn.Cell):
    def __init__(self, value):
        super(AssignAdd, self).__init__()
        self.var = Parameter(value, name="var")
        self.add = P.AssignAdd()

    def construct(self, y):
        self.add(self.var, y)
        return self.var


def assign_add(nptype):
    expect1 = np.array([[[[0, 2, 4.],
                          [6, 8, 10.],
                          [12, 14, 16.]],
                         [[18, 20, 22.],
                          [24, 26, 28.],
                          [30, 32, 34.]],
                         [[36, 38, 40.],
                          [42, 44, 46.],
                          [48, 50, 52.]]]])
    expect2 = np.array([[[[0, 3, 6],
                          [9, 12, 15],
                          [18, 21, 24]],
                         [[27, 30, 33],
                          [36, 39, 42],
                          [45, 48, 51]],
                         [[54, 57, 60],
                          [63, 66, 69],
                          [72, 75, 78]]]])
    x1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(nptype))
    y1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(nptype))

    x2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(nptype))
    y2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(nptype))

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    add = AssignAdd(x1)
    output1 = add(y1)
    assert (output1.asnumpy() == expect1).all()
    add = AssignAdd(output1)
    output2 = add(y1)
    assert (output2.asnumpy() == expect2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    add = AssignAdd(x2)
    output1 = add(y2)
    assert (output1.asnumpy() == expect1).all()
    add = AssignAdd(output1)
    output2 = add(y2)
    assert (output2.asnumpy() == expect2).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_fp16():
    """
    Feature: assign add kernel
    Description: test assignadd float16
    Expectation: just test
    """
    assign_add(np.float16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_fp32():
    """
    Feature: assign add kernel
    Description: test assignadd float32
    Expectation: just test
    """
    assign_add(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_fp64():
    """
    Feature: assign add kernel
    Description: test assignadd float64
    Expectation: just test
    """
    assign_add(np.float64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_comp64():
    """
    Feature: assign add kernel
    Description: test assignadd complex64
    Expectation: just test
    """
    assign_add(np.complex64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_comp128():
    """
    Feature: assign add kernel
    Description: test assignadd complex128
    Expectation: just test
    """
    assign_add(np.complex128)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_int8():
    """
    Feature: assign add kernel
    Description: test assignadd int8
    Expectation: just test
    """
    assign_add(np.int8)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_uint8():
    """
    Feature: assign add kernel
    Description: test assignadd uint8
    Expectation: just test
    """
    assign_add(np.uint8)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_int16():
    """
    Feature: assign add kernel
    Description: test assignadd int16
    Expectation: just test
    """
    assign_add(np.int16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_uint16():
    """
    Feature: assign add kernel
    Description: test assignadd uint16
    Expectation: just test
    """
    assign_add(np.uint16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_int32():
    """
    Feature: assign add kernel
    Description: test assignadd int32
    Expectation: just test
    """
    assign_add(np.int32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_uint32():
    """
    Feature: assign add kernel
    Description: test assignadd uint32
    Expectation: just test
    """
    assign_add(np.uint32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_int64():
    """
    Feature: assign add kernel
    Description: test assignadd int64
    Expectation: just test
    """
    assign_add(np.int64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_uint64():
    """
    Feature: assign add kernel
    Description: test assignadd uint64
    Expectation: just test
    """
    assign_add(np.uint64)
