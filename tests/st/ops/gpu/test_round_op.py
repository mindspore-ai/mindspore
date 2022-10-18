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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops, jit


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.round = ops.Round()

    def construct(self, x):
        return self.round(x)


def generate_testcases(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([0.9920, -0.4077, 0.9734, -1.0362, 1.5, -2.5, 4.5]).astype(nptype)
    net = Net()
    output = net(Tensor(x))
    expect = np.round(x).astype(nptype)
    np.testing.assert_almost_equal(output.asnumpy(), expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.array([0.9920, -0.4077, 0.9734, -1.0362, 1.5, -2.5, 4.5]).astype(nptype)
    net = Net()
    output = net(Tensor(x))
    expect = np.round(x).astype(nptype)
    np.testing.assert_almost_equal(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_round_float32():
    generate_testcases(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_round_float16():
    generate_testcases(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_round_functional():
    """
    Feature: functional round.
    Description: Test functional interface round.
    Expectation: success.
    """
    x = Tensor(np.array([1.1, 2.6, 4.5]), mindspore.float32)
    output = ops.round(x)
    assert np.all(output.asnumpy() == np.array([1, 3, 4]))


@jit
def round_fn_graph(x):
    return x.round()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_round_graph():
    """
    Feature: tensor round interface.
    Description: Test tensor round interface in graph mode.
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([1.1, 2.6, 4.5]), mindspore.float32)
    output = round_fn_graph(x)
    assert np.all(output.asnumpy() == np.array([1, 3, 4]))


def round_fn_pynative(x):
    return x.round()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_round_pynative():
    """
    Feature: tensor round interface.
    Description: Test tensor round interface in pynative mode.
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([1.1, 2.6, 4.5]), mindspore.float32)
    output = round_fn_pynative(x)
    assert np.all(output.asnumpy() == np.array([1, 3, 4]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_round_vmap():
    """
    Feature: vmap for ops Round.
    Description: Test operation Round with vmap.
    Expectation: success
    """
    x = Tensor(np.array([[1.1, 2.2], [3.3, 4.4]]), mindspore.float32)
    vmap_round = ops.vmap(round_fn_graph, 0, 1)
    output = vmap_round(x)
    assert np.all(output.asnumpy() == np.array([[1, 3], [2, 4]]))
