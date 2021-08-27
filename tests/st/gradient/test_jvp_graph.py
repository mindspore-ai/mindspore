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
"""test jvp in graph mode"""

import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn.grad import Jvp

context.set_context(mode=context.GRAPH_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3


class SingleInputMultipleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3, 2*x


class MultipleInputSingleOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2*x + 3*y


class MultipleInputMultipleOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2*x, y**3


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jvp_single_input_single_output_default_v_graph():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    primal, grad = Jvp(net)(x, v)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jvp_single_input_single_output_custom_v_graph():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 24], [81, 192]]).astype(np.float32))
    primal, grad = Jvp(net)(x, v)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jvp_single_input_multiple_outputs_default_v_graph():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    expect_primal_0 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    primal, grad = Jvp(net)(x, v)
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jvp_single_input_multiple_outputs_custom_v_graph():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    expect_primal_0 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[3, 24], [81, 192]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    primal, grad = Jvp(net)(x, v)
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jvp_multiple_inputs_single_output_default_v_graph():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    expect_primal = Tensor(np.array([[5, 10], [15, 20]]).astype(np.float32))
    expect_grad = Tensor(np.array([[5, 5], [5, 5]]).astype(np.float32))
    primal, grad = Jvp(net)(x, y, (v, v))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jvp_multiple_inputs_single_output_custom_v_graph():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v2 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    expect_primal = Tensor(np.array([[5, 10], [15, 20]]).astype(np.float32))
    expect_grad = Tensor(np.array([[5, 8], [11, 14]]).astype(np.float32))
    primal, grad = Jvp(net)(x, y, (v1, v2))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jvp_multiple_inputs_multiple_outputs_default_v_graph():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    expect_primal_0 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    primal, grad = Jvp(net)(x, y, (v, v))
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jvp_multiple_inputs_multiple_outputs_custom_v_graph():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v2 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    expect_primal_0 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[3, 24], [81, 192]]).astype(np.float32))
    primal, grad = Jvp(net)(x, y, (v1, v2))
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())
