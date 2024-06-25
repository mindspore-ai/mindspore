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
import numpy as np
import pytest
import mindspore.context as context
from mindspore import ops
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.unique = P.Unique()

    def construct(self, x):
        return self.unique(x)


class NetFunc(nn.Cell):
    def __init__(self):
        super(NetFunc, self).__init__()
        self.unique = ops.unique

    def construct(self, x):
        return self.unique(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unqiue():
    """
    Feature: Dynamic shape.
    Description: Test dynamic shape ops.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 3]), mstype.int32)
    unique = Net()
    output = unique(x)
    expect1 = np.array([1, 2, 3])
    expect2 = np.array([0, 0, 1, 1, 2, 2])
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unqiue_func_1d():
    """
    Feature: Test unique function
    Description: Input 1D Tensor
    Expectation: Successful execution.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 3]), mstype.int32)
    unique = NetFunc()
    output = unique(x)
    expect1 = np.array([1, 2, 3])
    expect2 = np.array([0, 0, 1, 1, 2, 2])
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unqiue_func_2d():
    """
    Feature: Test unique function
    Description: Input 2D Tensor
    Expectation: Successful execution.
    """
    x = Tensor(np.array([[1, 1, 2], [2, 3, 3]]), mstype.int32)
    unique = NetFunc()
    output = unique(x)
    expect1 = np.array([1, 2, 3])
    expect2 = np.array([[0, 0, 1], [1, 2, 2]])
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()
