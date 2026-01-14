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
"""Test cases for mint.nn.functional.conv1d operator."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import mint
from mindspore.mint.nn.functional import conv1d
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class Net1d(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mint_conv1d = mint.nn.functional.conv1d

    def construct(self, input_x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return self.mint_conv1d(input_x, weight, bias, stride, padding, dilation, groups)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=36.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_conv1d_default(mode):
    """
    Feature: mint.nn.functional.conv1d
    Description: Verify the result of conv1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    ## forward
    x = Tensor(np.linspace(0, 5, 1 * 4 * 8),
               ms.float32).reshape(1, 4, 8)
    bias = Tensor([0.1, 0.2], ms.float32)
    weight = Tensor(np.linspace(0, 5, 2 * 4 * 4),
                    ms.float32).reshape(2, 4, 4)
    net = Net1d()
    output = net(x, weight, bias=bias)
    expect_output = [[[59.41321, 62.534966, 65.65671, 68.77846, 71.90021],
                      [149.41956, 159.20103, 168.98251, 178.76399, 188.54547]]]
    assert np.allclose(output.asnumpy(), expect_output, atol=1e-4, rtol=1e-4)

    ## backward
    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias)
    expect_input_grad = [[[2.580645, 5.483871, 8.709678, 12.258064, 12.258064, 9.67742, 6.774194, 3.548387],
                          [3.8709679, 8.064516, 12.580645, 17.419353, 17.419353, 13.548388, 9.354839, 4.83871],
                          [5.16129, 10.645161, 16.451612, 22.580645, 22.580645, 17.419353, 11.935484, 6.129032],
                          [6.451613, 13.225806, 20.32258, 27.741936, 27.741936, 21.290323, 14.516129, 7.419355]]]
    expect_bias_grad = [5., 5.]
    expect_weight_grad = [[[1.6129031, 2.4193547, 3.2258065, 4.032258],
                           [8.064516, 8.870968, 9.67742, 10.483871],
                           [14.516129, 15.32258, 16.129032, 16.935484],
                           [20.967741, 21.774193, 22.580647, 23.387096]],
                          [[1.6129031, 2.4193547, 3.2258065, 4.032258],
                           [8.064516, 8.870968, 9.67742, 10.483871],
                           [14.516129, 15.32258, 16.129032, 16.935484],
                           [20.967741, 21.774193, 22.580647, 23.387096]]]

    assert np.allclose(grad_output[0].asnumpy(), expect_input_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[1].asnumpy(), expect_weight_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[2].asnumpy(), expect_bias_grad, atol=1e-4, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=36.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_conv1d_pad_mode_same(mode):
    """
    Feature: mint.nn.functional.conv1d
    Description: Verify the result of conv1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    ## forward
    x = Tensor(np.linspace(0, 5, 1 * 4 * 8),
               ms.float32).reshape(1, 4, 8)
    bias = Tensor([0.1, 0.2], ms.float32)
    weight = Tensor(np.linspace(0, 5, 2 * 4 * 4),
                    ms.float32).reshape(2, 4, 4)
    net = Net1d()
    output = net(x, weight, bias=bias, padding="same")
    expect_output = [[[45.26129, 59.41321, 62.534966, 65.65671, 68.77846, 71.90021, 52.129135, 33.502705],
                      [110.29365, 149.41956, 159.20103, 168.98251, 178.76399, 188.54547, 142.13548, 95.2052]]]
    assert np.allclose(output.asnumpy(), expect_output, atol=1e-4, rtol=1e-4)

    ## backward
    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias)
    expect_input_grad = [[[2.580645, 5.483871, 8.709678, 12.258064, 12.258064, 9.67742, 6.774194, 3.548387],
                          [3.8709679, 8.064516, 12.580645, 17.419353, 17.419353, 13.548388, 9.354839, 4.83871],
                          [5.16129, 10.645161, 16.451612, 22.580645, 22.580645, 17.419353, 11.935484, 6.129032],
                          [6.451613, 13.225806, 20.32258, 27.741936, 27.741936, 21.290323, 14.516129, 7.419355]]]
    expect_bias_grad = [5., 5.]
    expect_weight_grad = [[[1.6129031, 2.4193547, 3.2258065, 4.032258],
                           [8.064516, 8.870968, 9.67742, 10.483871],
                           [14.516129, 15.32258, 16.129032, 16.935484],
                           [20.967741, 21.774193, 22.580647, 23.387096]],
                          [[1.6129031, 2.4193547, 3.2258065, 4.032258],
                           [8.064516, 8.870968, 9.67742, 10.483871],
                           [14.516129, 15.32258, 16.129032, 16.935484],
                           [20.967741, 21.774193, 22.580647, 23.387096]]]

    assert np.allclose(grad_output[0].asnumpy(), expect_input_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[1].asnumpy(), expect_weight_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[2].asnumpy(), expect_bias_grad, atol=1e-4, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_conv1d_bfloat16(mode):
    """
    Feature: mint.nn.functional.conv1d
    Description: Verify the result of conv1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    ## forward
    x = Tensor(np.linspace(0, 5, 1 * 4 * 8),
               ms.bfloat16).reshape(1, 4, 8)
    bias = Tensor([0.1, 0.2], ms.bfloat16)
    weight = Tensor(np.linspace(0, 5, 2 * 4 * 4),
                    ms.bfloat16).reshape(2, 4, 4)
    net = Net1d()
    output = net(x, weight, bias=bias, padding="valid")
    expect_output = [[[59.41321, 62.534966, 65.65671, 68.77846, 71.90021],
                      [149.41956, 159.20103, 168.98251, 178.76399, 188.54547]]]
    assert np.allclose(output.float().asnumpy(), expect_output, atol=4e-3, rtol=4e-3)

    ## backward
    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias)
    expect_input_grad = [[[2.580645, 5.483871, 8.709678, 12.258064, 12.258064, 9.67742, 6.774194, 3.548387],
                          [3.8709679, 8.064516, 12.580645, 17.419353, 17.419353, 13.548388, 9.354839, 4.83871],
                          [5.16129, 10.645161, 16.451612, 22.580645, 22.580645, 17.419353, 11.935484, 6.129032],
                          [6.451613, 13.225806, 20.32258, 27.741936, 27.741936, 21.290323, 14.516129, 7.419355]]]
    expect_bias_grad = [5., 5.]
    expect_weight_grad = [[[1.6129031, 2.4193547, 3.2258065, 4.032258],
                           [8.064516, 8.870968, 9.67742, 10.483871],
                           [14.516129, 15.32258, 16.129032, 16.935484],
                           [20.967741, 21.774193, 22.580647, 23.387096]],
                          [[1.6129031, 2.4193547, 3.2258065, 4.032258],
                           [8.064516, 8.870968, 9.67742, 10.483871],
                           [14.516129, 15.32258, 16.129032, 16.935484],
                           [20.967741, 21.774193, 22.580647, 23.387096]]]

    assert np.allclose(grad_output[0].float().asnumpy(), expect_input_grad, atol=4e-3, rtol=4e-3)
    assert np.allclose(grad_output[1].float().asnumpy(), expect_weight_grad, atol=4e-3, rtol=4e-3)
    assert np.allclose(grad_output[2].float().asnumpy(), expect_bias_grad, atol=4e-3, rtol=4e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=36.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_conv1d_batchfy(mode):
    """
    Feature: mint.nn.functional.conv1d
    Description: Verify the result of conv1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    ## forward
    x = Tensor(np.linspace(0, 5, 4 * 8),
               ms.float32).reshape(4, 8)
    bias = Tensor([0.1, 0.2], ms.float32)
    weight = Tensor(np.linspace(0, 5, 2 * 4 * 4),
                    ms.float32).reshape(2, 4, 4)
    net = Net1d()
    output = net(x, weight, bias=bias)
    expect_output = [[59.41321, 62.534966, 65.65671, 68.77846, 71.90021],
                     [149.41956, 159.20103, 168.98251, 178.76399, 188.54547]]
    assert np.allclose(output.asnumpy(), expect_output, atol=1e-4, rtol=1e-4)

    ## backward
    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias)
    expect_input_grad = [[2.580645, 5.483871, 8.709678, 12.258064, 12.258064, 9.67742, 6.774194, 3.548387],
                         [3.8709679, 8.064516, 12.580645, 17.419353, 17.419353, 13.548388, 9.354839, 4.83871],
                         [5.16129, 10.645161, 16.451612, 22.580645, 22.580645, 17.419353, 11.935484, 6.129032],
                         [6.451613, 13.225806, 20.32258, 27.741936, 27.741936, 21.290323, 14.516129, 7.419355]]
    expect_bias_grad = [5., 5.]
    expect_weight_grad = [[[1.6129031, 2.4193547, 3.2258065, 4.032258],
                           [8.064516, 8.870968, 9.67742, 10.483871],
                           [14.516129, 15.32258, 16.129032, 16.935484],
                           [20.967741, 21.774193, 22.580647, 23.387096]],
                          [[1.6129031, 2.4193547, 3.2258065, 4.032258],
                           [8.064516, 8.870968, 9.67742, 10.483871],
                           [14.516129, 15.32258, 16.129032, 16.935484],
                           [20.967741, 21.774193, 22.580647, 23.387096]]]

    assert np.allclose(grad_output[0].asnumpy(), expect_input_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[1].asnumpy(), expect_weight_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[2].asnumpy(), expect_bias_grad, atol=1e-4, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_conv1d_dynamic():
    """
    Feature: mint.nn.functional.conv2d
    Description: dynamic shape and rank
    Expectation: success
    """
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    x1 = ms.Tensor(np.ones([2, 2, 4]), ms.float16)
    weight1 = ms.Tensor(np.ones([2, 2, 1]), ms.float16)
    x2 = ms.Tensor(np.ones([1, 2, 6]), ms.float16)
    weight2 = ms.Tensor(np.ones([2, 2, 2]), ms.float16)
    bias = ms.Tensor(np.ones([2]), ms.float16)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    TEST_OP(conv1d, [[x1, weight1, bias, stride, padding, dilation, groups],
                     [x2, weight2, bias, stride, padding, dilation, groups]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})
