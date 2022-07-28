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

import pytest
import numpy as np
import mindspore.context as context
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.functional import vmap


class FuncNet(nn.Cell):
    def __init__(self, paddings):
        super(FuncNet, self).__init__()
        self.paddings = paddings

    def construct(self, x):
        return ops.pad(x, self.paddings)


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.network = network
        self.grad = ops.GradOperation()

    def construct(self, x):
        return self.grad(self.network)(x)


def run_case(x, paddings, expect):
    net = FuncNet(paddings)
    out_ms = net(Tensor(x))
    assert np.allclose(expect, out_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pad_function_cpu():
    """
    Feature: test ops.Pad functional interface.
    Description: paddings has negative values.
    Expectation: the result match with numpy result.
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    # case1: padding value are non negative
    paddings1 = ((0, 1), (1, 0))
    expect1 = np.array([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9], [0, 0, 0, 0]], dtype=np.float32)
    # case2: padding value are non positive
    paddings2 = ((-1, 0), (-1, -1))
    expect2 = np.array([[5], [8]], dtype=np.float32)
    # case3: padding with positive and negative value
    paddings3 = ((-1, 1), (1, -1))
    expect3 = np.array([[0, 4, 5], [0, 7, 8], [0, 0, 0]], dtype=np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_case(x, paddings1, expect1)
    run_case(x, paddings2, expect2)
    run_case(x, paddings3, expect3)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    run_case(x, paddings1, expect1)
    run_case(x, paddings2, expect2)
    run_case(x, paddings3, expect3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pad_function_grad_cpu():
    """
    Feature: test ops.Pad functional interface backward.
    Description: paddings has negative values.
    Expectation: the result match with numpy result.
    """
    paddings = ((1, -1), (1, -1))
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    expect = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = GradNet(FuncNet(paddings))
    out_ms = net(Tensor(x))
    assert np.allclose(expect, out_ms.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    net = GradNet(FuncNet(paddings))
    out_ms = net(Tensor(x))
    assert np.allclose(expect, out_ms.asnumpy())


def vmap_case():
    class Net(nn.Cell):
        def __init__(self, paddings):
            super(Net, self).__init__()
            self.pad = ops.Pad(paddings)

        def construct(self, x):
            return self.pad(x)

    # single vmap case
    x_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    expect = np.array([[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]], dtype=np.float32)
    out_ms = vmap(Net(((1, 1),)), 0, 0)(Tensor(x_np))
    assert np.allclose(expect, out_ms.asnumpy())
    # nested vmap case
    x_np1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    expect1 = np.array([[[0, 1, 2, 0], [0, 3, 4, 0]], [[0, 5, 6, 0], [0, 7, 8, 0]]], dtype=np.float32)
    out_ms1 = vmap(vmap(Net(((1, 1),)), 0, 0), 0, 0)(Tensor(x_np1))
    assert np.allclose(expect1, out_ms1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pad_vmap_cpu():
    """
    Feature: test ops.Pad vmap.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    vmap_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    vmap_case()
