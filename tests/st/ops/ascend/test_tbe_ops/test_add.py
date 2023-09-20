# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class AddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


class AddDynamicShapeNet(nn.Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.add = P.Add()
        self.axis = axis

    def construct(self, x_, y_, indices):
        u_indices, _ = self.unique(indices)
        x_ = self.gather(x_, u_indices, self.axis)
        y_ = self.gather(y_, u_indices, self.axis)
        return self.add(x_, y_)


def comput_expect(x, y):
    return np.add(x, y)


def add_net(*args, is_dynamic=False):
    op = args[0]
    x = args[1]
    y = args[2]
    if is_dynamic:
        out = op(Tensor(x), Tensor(y), Tensor(args[3]))
    else:
        out = op(Tensor(x), Tensor(y))
    if is_dynamic:
        print("input shape: ", x.shape)
        print("output shape: ", out.shape)
    else:
        assert np.allclose(out.asnumpy(), comput_expect(x, y), 1e-3, 1e-3)


@pytest.mark.skip
def test_add(dtype=np.float16):
    """
    Feature: test add operator in graph and pynative mode.
    Description: test add.
    Expectation: the result is correct
    """
    x = np.random.randn(3, 3, 4).astype(dtype)
    y = np.random.randn(3, 3, 4).astype(dtype)
    indices = np.random.randint(0, 3, size=3)

    net = AddNet()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    add_net(net, x, y)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    add_net(net, x, y)

    net = AddDynamicShapeNet()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    add_net(net, x, y, indices, is_dynamic=True)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    add_net(net, x, y, indices, is_dynamic=True)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_add_float16():
    """
    Feature: test add operator.
    Description: test float16 input.
    Expectation: the result is correct
    """
    test_add(np.float16)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_add_float32():
    """
    Feature: test add operator.
    Description: test float32 input.
    Expectation: the result is correct
    """
    test_add(np.float32)
