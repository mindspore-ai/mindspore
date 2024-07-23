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
from mindspore import Tensor
from mindspore.ops.operations.sparse_ops import SparseToDenseV2
from mindspore.common.api import jit
import mindspore.common.dtype as mstype


class SparseToDenseNet(nn.Cell):
    def __init__(self):
        super(SparseToDenseNet, self).__init__()
        self.sparsetodense = SparseToDenseV2()

    @jit
    def construct(self, indices, output_shape, values, default_value):
        return self.sparsetodense(indices, output_shape, values, default_value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparsetodense_2d_int32():
    """
    Feature: Converts a sparse representation into a dense tensor.
    Description: 2D , int32
    Expectation: success
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices = Tensor(np.array([[0, 1]]).astype(np.int32))
        output_shape = Tensor(np.array([2, 2]).astype(np.int32))
        values = Tensor(np.array([1]).astype(np.int32))
        default_value = Tensor(0, dtype=mstype.int32)
        net = SparseToDenseNet()
        output = net(indices, output_shape, values, default_value)
        sparse_expect = np.array([[0, 1],
                                  [0, 0]]).astype(np.int32)
        assert (output.asnumpy() == sparse_expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparsetodense_2d_double():
    """
    Feature: Converts a sparse representation into a dense tensor.
    Description: 2D , double
    Expectation: success
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices = Tensor(np.array([[0, 1]]).astype(np.int32))
        output_shape = Tensor(np.array([2, 2]).astype(np.int32))
        values = Tensor(np.array([1.0]).astype(np.double))
        default_value = Tensor(0.0, dtype=mstype.double)
        net = SparseToDenseNet()
        output = net(indices, output_shape, values, default_value)
        sparse_expect = np.array([[0.0, 1.0],
                                  [0.0, 0.0]]).astype(np.double)
        assert (output.asnumpy() == sparse_expect).all()
