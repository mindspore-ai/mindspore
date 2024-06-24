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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops.operations import array_ops as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bincount = P.Bincount()

    def construct(self, array, size, weights):
        return self.bincount(array, size, weights)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bincount_graph():
    """
    Feature: Bincount
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    types = [mstype.float32, mstype.float64, mstype.int32, mstype.int64]
    for type_i in types:
        input_array = Tensor(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]), mstype.int32)
        input_size = Tensor(5, mstype.int32)
        input_weights = Tensor(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), type_i)
        net = Net()
        output = net(input_array, input_size, input_weights).asnumpy()
        expect = np.array([0, 1, 2, 3, 4]).astype(np.float32)
        assert np.allclose(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bincount_pynative():
    """
    Feature: Bincount
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    types = [mstype.float32, mstype.float64, mstype.int32, mstype.int64]
    for type_i in types:
        input_array = Tensor(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]), mstype.int32)
        input_size = Tensor(5, mstype.int32)
        input_weights = Tensor(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), type_i)
        net = Net()
        output = net(input_array, input_size, input_weights).asnumpy()
        expect = np.array([0, 1, 2, 3, 4]).astype(np.float32)
        assert np.allclose(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bincount_bigdata():
    """
    Feature: Bincount
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_array = Tensor(np.array([4, 4, 6, 67, 6, 9, 6, 0, 0, 56, 3, 9]), mstype.int32)
    input_size = Tensor(6, mstype.int32)
    input_weights = Tensor(np.array([45423, 5135415, 56845696, 656795677, 234523478, 204354719, 25001235166,
                                     41345, 264, 5563566, 3756, 976575]), mstype.int64)
    net = Net()
    output = net(input_array, input_size, input_weights).asnumpy()
    expect = np.array([41609, 0, 0, 3756, 5180838, 0]).astype(np.int64)
    assert np.allclose(output, expect)
