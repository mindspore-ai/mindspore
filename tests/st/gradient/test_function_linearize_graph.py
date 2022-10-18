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
"""test function linearize in graph mode"""

import numpy as np
import pytest
from mindspore import nn
from mindspore import context
from mindspore import Tensor
from mindspore import jit
from mindspore.ops.functional import linearize, jvp

context.set_context(mode=context.GRAPH_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3


class SingleInputMultipleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3, 2 * x


class MultipleInputSingleOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2 * x + 3 * y


class MultipleInputMultipleOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2 * x, y**3


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linearize_single_input_single_output_diverse_v_graph():
    """
    Features: Function linearize
    Description: Test linearize with single input, single output and linearize v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v_0 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v_1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_primal, expect_grad_0 = jvp(net, x, v_0)
    expect_primal, expect_grad_1 = jvp(net, x, v_1)
    primal, jvp_fn = linearize(net, x)
    grad_0 = jvp_fn(v_0)
    grad_1 = jvp_fn(v_1)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad_0.asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad_1.asnumpy(), expect_grad_1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linearize_single_input_multiple_outputs_diverse_v_graph():
    """
    Features: Function linearize
    Description: Test linearize with single input, multiple outputs and linearize v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v_0 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v_1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    expect_primal, expect_grad_0 = jvp(net, x, v_0)
    expect_primal, expect_grad_1 = jvp(net, x, v_1)

    primal, jvp_fn = linearize(net, x)
    grad_0 = jvp_fn(v_0)
    grad_1 = jvp_fn(v_1)
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal[0].asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal[1].asnumpy())
    assert isinstance(grad_0, tuple)
    assert len(grad_0) == 2
    assert np.allclose(grad_0[0].asnumpy(), expect_grad_0[0].asnumpy())
    assert np.allclose(grad_0[1].asnumpy(), expect_grad_0[1].asnumpy())
    assert isinstance(grad_1, tuple)
    assert len(grad_1) == 2
    assert np.allclose(grad_1[0].asnumpy(), expect_grad_1[0].asnumpy())
    assert np.allclose(grad_1[1].asnumpy(), expect_grad_1[1].asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linearize_multiple_inputs_single_output_diverse_v_graph():
    """
    Features: Function linearize
    Description: Test linearize with multiple inputs, single output and diverse v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v_0 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v_1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    expect_primal, expect_grad_0 = jvp(net, (x, y), (v_0, v_0))
    expect_primal, expect_grad_1 = jvp(net, (x, y), (v_0, v_1))
    primal, jvp_fn = linearize(net, (x, y))
    grad_0 = jvp_fn((v_0, v_0))
    grad_1 = jvp_fn((v_0, v_1))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad_0.asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad_1.asnumpy(), expect_grad_1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linearize_multiple_inputs_multiple_outputs_diverse_v_graph():
    """
    Features: Function linearize
    Description: Test linearize with multiple inputs, multiple outputs and diverse v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v_0 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v_1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    expect_primal, expect_grad_0 = jvp(net, (x, y), (v_0, v_0))
    expect_primal, expect_grad_1 = jvp(net, (x, y), (v_0, v_1))
    primal, jvp_fn = linearize(net, (x, y))
    grad_0 = jvp_fn((v_0, v_0))
    grad_1 = jvp_fn((v_0, v_1))
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal[0].asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal[1].asnumpy())
    assert isinstance(grad_0, tuple)
    assert len(grad_0) == 2
    assert np.allclose(grad_0[0].asnumpy(), expect_grad_0[0].asnumpy())
    assert np.allclose(grad_0[1].asnumpy(), expect_grad_0[1].asnumpy())
    assert isinstance(grad_1, tuple)
    assert len(grad_1) == 2
    assert np.allclose(grad_1[0].asnumpy(), expect_grad_1[0].asnumpy())
    assert np.allclose(grad_1[1].asnumpy(), expect_grad_1[1].asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linearize_input_function_single_input_single_output_diverse_v_graph():
    """
    Features: Function linearize
    Description: Test linearize with function, single input, single output and default v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v_0 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v_1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))

    def test_function(inputs):
        return inputs**3

    expect_primal, expect_grad_0 = jvp(test_function, x, v_0)
    expect_primal, expect_grad_1 = jvp(test_function, x, v_1)
    primal, jvp_fn = linearize(test_function, x)
    grad_0 = jvp_fn(v_0)
    grad_1 = jvp_fn(v_1)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad_0.asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad_1.asnumpy(), expect_grad_1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linearize_jit_function_single_input_single_output_diverse_v_graph():
    """
    Features: Function linearize
    Description: Test linearize with @jit decorated function, single input, single output and diverse v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v_0 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v_1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()

    @jit
    def linearize_with_jit_function(inputs, v_0, v_1):
        output, jvp_fn = linearize(net, inputs)
        grad_0 = jvp_fn(v_0)
        grad_1 = jvp_fn(v_1)
        return output, grad_0, grad_1

    expect_primal, expect_grad_0 = jvp(net, x, v_0)
    expect_primal, expect_grad_1 = jvp(net, x, v_1)
    primal, grad_0, grad_1 = linearize_with_jit_function(x, v_0, v_1)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad_0.asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad_1.asnumpy(), expect_grad_1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linearize_construct_single_input_single_output_diverse_v_graph():
    """
    Features: Function linearize
    Description: Test linearize with construct, single input, single output and diverse v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v_0 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v_1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()

    class Net(nn.Cell):
        def __init__(self, network):
            super(Net, self).__init__()
            self.net = network

        def construct(self, inputs, v_0, v_1):
            output, jvp_fn = linearize(net, inputs)
            grad_0 = jvp_fn(v_0)
            grad_1 = jvp_fn(v_1)
            return output, grad_0, grad_1

    test_net = Net(SingleInputSingleOutputNet())
    expect_primal, expect_grad_0 = jvp(net, x, v_0)
    expect_primal, expect_grad_1 = jvp(net, x, v_1)
    primal, grad_0, grad_1 = test_net(x, v_0, v_1)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad_0.asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad_1.asnumpy(), expect_grad_1.asnumpy())
