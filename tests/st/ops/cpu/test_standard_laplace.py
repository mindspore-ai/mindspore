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

import pytest
import numpy as np

import mindspore.context as context
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetStandardLaplace(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(NetStandardLaplace, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.stdlaplace = ops.StandardLaplace(seed, seed2)

    def construct(self):
        return self.stdlaplace(self.shape)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_standard_laplace_op():
    """
    Feature: StandardLaplace CPU operation
    Description: input the shape and random seed, test the output value and shape
    Expectation: the value and shape of output tensor match the predefined values
    """
    seed = 10
    seed2 = 10
    shape = (5, 6, 8)
    net = NetStandardLaplace(shape, seed, seed2)
    output = net()
    assert output.shape == (5, 6, 8)
    outnumpyflatten_1 = output.asnumpy().flatten()

    seed = 10
    seed2 = 10
    shape = (5, 6, 8)
    net = NetStandardLaplace(shape, seed, seed2)
    output = net()
    assert output.shape == (5, 6, 8)
    outnumpyflatten_2 = output.asnumpy().flatten()
    # same seed should generate same random number
    assert (outnumpyflatten_1 == outnumpyflatten_2).all()

    seed = 0
    seed2 = 0
    shape = (130, 120, 141)
    net = NetStandardLaplace(shape, seed, seed2)
    output = net()
    assert output.shape == (130, 120, 141)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_standard_laplace_functional():
    """
    Feature: Functional interface of StandardLaplace CPU operation
    Description: input the shape and random seed, test the output value and shape
    Expectation: the value and shape of output tensor match the predefined values
    """
    seed = 8
    shape = (5, 6, 8)
    output = ops.standard_laplace(shape, seed)
    assert output.shape == shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_standard_laplace_dynamic_shape():
    """
    Feature: Dynamic shape inference of StandardLaplace CPU operation
    Description: input a dynamic shape, test the output shape
    Expectation: the shape of output match the input shape Tensor
    """
    class DynamicShapeStandardLaplaceNet(nn.Cell):
        def __init__(self, axis=0):
            super().__init__()
            self.unique = ops.Unique()
            self.gather = ops.Gather()
            self.get_shape = ops.TensorShape()
            self.random_op = ops.StandardLaplace()
            self.axis = axis

        def construct(self, x, indices):
            unique_indices, _ = self.unique(indices)
            res = self.gather(x, unique_indices, self.axis)
            dshape = self.get_shape(res)
            return self.random_op(dshape), dshape
    net = DynamicShapeStandardLaplaceNet()
    input_x = Tensor(np.random.randint(1, 10, size=10))
    indices_x = Tensor(np.random.randint(1, 10, size=7))
    out, dshape = net(input_x, indices_x)
    assert out.shape == tuple(dshape.asnumpy())
