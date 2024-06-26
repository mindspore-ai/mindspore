# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops.auto_generate import LayerNormV3
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.layernorm = LayerNormV3()

    def construct(self, input_x, gamma, beta):
        return self.layernorm(input_x, gamma, beta)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_layer_norm_v3(mode):
    """
    Feature: test LayerNormV3 forward.
    Description: test LayerNormV3 inputs.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
    gamma = Tensor(np.ones([3]), mindspore.float32)
    beta = Tensor(np.ones([3]), mindspore.float32)
    net = Net()
    output, mean, variance = net(input_x, gamma, beta)

    expect_output = np.array([[-0.22474468, 1., 2.22474468], [-0.22474468, 1., 2.22474468]])
    expect_mean = np.array([[2.], [2.]])
    expect_var = np.array([[1.2247447], [1.2247447]])

    assert np.allclose(output.asnumpy(), expect_output, atol=1e-6)
    assert np.allclose(mean.asnumpy(), expect_mean, atol=1e-6)
    assert np.allclose(variance.asnumpy(), expect_var, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_layer_norm_grad_v3(mode):
    """
    Feature: test LayerNormV3 backward.
    Description: test LayerNormGradV3.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
    gamma = Tensor(np.ones([3]), mindspore.float32)
    beta = Tensor(np.ones([3]), mindspore.float32)
    net = Net()

    grads = ops.grad(net, (1))(input_x, gamma, beta)
    except_grads = np.array([-2.4494894, 0., 2.4494894])
    assert np.allclose(grads.asnumpy(), except_grads, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_layer_norm_v3_dynamic_shape(mode):
    """
    Feature: test LayerNormV3 forward dynamic shape.
    Description: test LayerNormV3 inputs.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
    gamma = Tensor(np.ones([3]), mindspore.float32)
    beta = Tensor(np.ones([3]), mindspore.float32)
    net = Net()
    input_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    net.set_inputs(input_dyn, gamma, beta)
    output, mean, variance = net(input_x, gamma, beta)

    expect_output = np.array([[-0.22474468, 1., 2.22474468], [-0.22474468, 1., 2.22474468]])
    expect_mean = np.array([[2.], [2.]])
    expect_var = np.array([[1.2247447], [1.2247447]])

    assert np.allclose(output.asnumpy(), expect_output, atol=1e-6)
    assert np.allclose(mean.asnumpy(), expect_mean, atol=1e-6)
    assert np.allclose(variance.asnumpy(), expect_var, atol=1e-6)

    input_x2 = Tensor(np.array([[2, 4, 6], [5, 7, 9]]), mindspore.float32)
    output2, mean2, variance2 = net(input_x2, gamma, beta)
    expect_output2 = np.array([[-0.2247448, 1., 2.2247448], [-0.2247448, 1., 2.2247448]])
    expect_mean2 = np.array([[4.], [7.]])
    expect_var2 = np.array([[0.6123724], [0.6123724]])

    assert np.allclose(output2.asnumpy(), expect_output2, atol=1e-6)
    assert np.allclose(mean2.asnumpy(), expect_mean2, atol=1e-6)
    assert np.allclose(variance2.asnumpy(), expect_var2, atol=1e-6)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_layer_norm_grad_v3_dynamic_shape(mode):
    """
    Feature: test LayerNormV3 backward dynamic shape.
    Description: test LayerNormGradV3.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
    gamma = Tensor(np.ones([3]), mindspore.float32)
    beta = Tensor(np.ones([3]), mindspore.float32)
    net = Net()
    input_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    net.set_inputs(input_dyn, gamma, beta)

    grads = ops.grad(net, (1))(input_x, gamma, beta)
    except_grads = np.array([-2.4494894, 0., 2.4494894])
    assert np.allclose(grads.asnumpy(), except_grads, atol=1e-6)

    input_x2 = Tensor(np.array([[1, 4, 6], [2, 7, 9]]), mindspore.float32)
    grads2 = ops.grad(net, (1))(input_x2, gamma, beta)
    except_grads2 = np.array([-2.6565037, 0.5019045, 2.1545992])
    assert np.allclose(grads2.asnumpy(), except_grads2, atol=1e-6)
