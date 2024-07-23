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
import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.Dropout()

    def construct(self, x_):
        return self.op(x_)


class DynamicShapeNet(nn.Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.op = P.Dropout()
        self.axis = axis

    def construct(self, x_, indices):
        u_indices, _ = self.unique(indices)
        x_ = self.gather(x_, u_indices, self.axis)
        return self.op(x_)


def dropout_net(*args, is_dynamic=False):
    op = args[0]
    x = args[1]
    if is_dynamic:
        out = op(Tensor(x), Tensor(args[2]))
    else:
        out = op(Tensor(x))
    print("input shape: ", x.shape)
    print("output shape: ", out[0].shape)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dropout_bf16():
    """
    Feature: test dropout operator in graph and pynative mode.
    Description: test dropout.
    Expectation: the result is correct
    """
    x = np.random.randn(3, 3, 4).astype(np.float32)
    net = Net()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    out = net(Tensor(x, dtype=mindspore.bfloat16))
    print("output shape: ", out[0].shape)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    out = net(Tensor(x, dtype=mindspore.bfloat16))
    print("output shape: ", out[0].shape)


@pytest.mark.skip
def test_dropout(dtype=np.float16):
    """
    Feature: test dropout operator in graph and pynative mode.
    Description: test dropout.
    Expectation: the result is correct
    """
    x = np.random.randn(3, 3, 4).astype(dtype)
    indices = np.random.randint(0, 3, size=3)

    net = Net()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dropout_net(net, x)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dropout_net(net, x)

    net = DynamicShapeNet()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dropout_net(net, x, indices, is_dynamic=True)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dropout_net(net, x, indices, is_dynamic=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_float16():
    """
    Feature: test dropout operator.
    Description: test float16 input.
    Expectation: the result is correct
    """
    test_dropout(np.float16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_float32():
    """
    Feature: test dropout operator.
    Description: test float32 input.
    Expectation: the result is correct
    """
    test_dropout(np.float32)
