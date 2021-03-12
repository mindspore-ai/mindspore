# Copyright 2021 Huawei Technologies Co., Ltd
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


import numpy as np
import pytest

from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.context as context


class PrintNetOneInput(nn.Cell):
    def __init__(self):
        super(PrintNetOneInput, self).__init__()
        self.op = P.Print()

    def construct(self, x):
        self.op(x)
        return x


class PrintNetTwoInputs(nn.Cell):
    def __init__(self):
        super(PrintNetTwoInputs, self).__init__()
        self.op = P.Print()

    def construct(self, x, y):
        self.op(x, y)
        return x


class PrintNetIndex(nn.Cell):
    def __init__(self):
        super(PrintNetIndex, self).__init__()
        self.op = P.Print()

    def construct(self, x):
        self.op(x[0][0][6][3])
        return x


def print_testcase(nptype):
    # large shape
    x = np.arange(20808).reshape(6, 3, 34, 34).astype(nptype)
    # a value that can be stored as int8_t
    x[0][0][6][3] = 125
    # small shape
    y = np.arange(9).reshape(3, 3).astype(nptype)
    x = Tensor(x)
    y = Tensor(y)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net_1 = PrintNetOneInput()
    net_2 = PrintNetTwoInputs()
    net_3 = PrintNetIndex()
    net_1(x)
    net_2(x, y)
    net_3(x)

class PrintNetString(nn.Cell):
    def __init__(self):
        super(PrintNetString, self).__init__()
        self.op = P.Print()

    def construct(self, x, y):
        self.op("The first Tensor is", x)
        self.op("The second Tensor is", y)
        self.op("This line only prints string", "Another line")
        self.op("The first Tensor is", x, y, "is the second Tensor")
        return x

def print_testcase_string(nptype):
    x = np.ones(18).astype(nptype)
    y = np.arange(9).reshape(3, 3).astype(nptype)
    x = Tensor(x)
    y = Tensor(y)
    # graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = PrintNetString()
    net(x, y)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_bool():
    print_testcase(np.bool)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_int8():
    print_testcase(np.int8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_int16():
    print_testcase(np.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_int32():
    print_testcase(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_int64():
    print_testcase(np.int64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_uint8():
    print_testcase(np.uint8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_uint16():
    print_testcase(np.uint16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_uint32():
    print_testcase(np.uint32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_uint64():
    print_testcase(np.uint64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_float16():
    print_testcase(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_float32():
    print_testcase(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_print_string():
    print_testcase_string(np.float32)
