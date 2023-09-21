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
"""
test assign sub
"""
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.dtype import pytype_to_dtype
from mindspore import Tensor

import pytest
import numpy as np



class AssignW(nn.Cell):
    def __init__(self):
        super(AssignW, self).__init__()
        self.assign = P.Assign()

    def construct(self, x, w):
        self.assign(x, w)
        return x


class AssignOp(nn.Cell):
    def __init__(self):
        super(AssignOp, self).__init__()
        self.b = Parameter(initializer('ones', [5]), name='b')

    def construct(self, w):
        self.b = w
        return w


def test_assign_by_operator():
    context.set_context(mode=context.GRAPH_MODE)
    net = AssignOp()
    net.to_float(ms.float16)
    input_data = Tensor(np.ones([5]).astype(np.float32))
    net(input_data)


class NetScatterNdUpdate(nn.Cell):
    def __init__(self):
        super(NetScatterNdUpdate, self).__init__()
        self.b = Parameter(initializer('ones', [5, 5]), name='b')
        self.scatter = P.ScatterNdUpdate()

    def construct(self, idx, x):
        return self.scatter(self.b, idx, x)


def test_scatter_nd_update():
    context.set_context(mode=context.GRAPH_MODE)
    net = NetScatterNdUpdate()
    x = Tensor(np.ones([5]).astype(np.float16))
    idx = Tensor(np.ones([1]).astype(np.int32))
    with pytest.raises(ValueError) as ex:
        net(idx, x)
        assert "the dimension of \'indices\' must be greater than or equal to 2" in str(ex.value)


def test_signature_error_info():
    '''
    Feature: Do signature error info.
    Description:  Do signature error info.
    Expectation: RuntimeError
    '''
    class NetScatterDiv(nn.Cell):
        def __init__(self):
            super(NetScatterDiv, self).__init__()
            self.b = Parameter(initializer(1, [5, 3], pytype_to_dtype(np.int8)), name='input')
            self.scatter = P.ScatterDiv(False)

        def construct(self, idx, x):
            return self.scatter(self.b, idx, x)

    net = NetScatterDiv()
    with pytest.raises(TypeError) as ex:
        net(Tensor(np.random.randint(1, size=(5, 8, 2)).astype(np.int32)),
            Tensor(np.random.randint(1, 256, size=(5, 8, 2, 3)).astype(np.float32)))
    assert "Data type conversion of \'Parameter\' is not supported, " \
           "the argument[x]'s data type of primitive[ScatterDiv] is int8, " \
           "which cannot be converted to data type float32 automatically." in str(ex.value)
