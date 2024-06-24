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

import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, reduction=None):
        super(Net, self).__init__()
        if reduction is not None:
            self.kl_div_loss_grad = G.KLDivLossGrad(reduction)
        else:
            self.kl_div_loss_grad = G.KLDivLossGrad()

    def construct(self, x, y, dy):
        return self.kl_div_loss_grad(dy, x, y)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mode_none_and_dtype_with_static_input(dtype):
    """
    Feature: KLDivLossGrad with none reduction mode.
    Description: KLDivLossGrad with none reduction mode, 2d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = mindspore.Tensor(np.array([[-1, 0], [1, 1]]).astype(dtype))
    net = Net("none")
    output = net(prediction, target, dy)
    print(output)
    expect = np.array([[0, 0], [-1, 0]])
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mode_mean_and_dtype_with_static_input(dtype):
    """
    Feature: KLDivLossGrad with mean reduction mode.
    Description: KLDivLossGrad with mean reduction mode, 2d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = mindspore.Tensor(np.array([-1]).astype(dtype))
    net = Net("mean")
    output = net(prediction, target, dy)
    print(output)
    expect = np.array([[0, 0.25], [0.25, 0]]).astype(dtype)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mode_sum_and_dtype_with_static_input(dtype):
    """
    Feature: KLDivLossGrad with sum reduction mode.
    Description: KLDivLossGrad with sum reduction mode, 2d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = mindspore.Tensor(np.array([-1]).astype(dtype))
    net = Net("sum")
    output = net(prediction, target, dy)
    print(output)
    expect = np.array([[0, 1], [1, 0]])
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mode_batchmean_and_dtype_with_static_input(dtype):
    """
    Feature: KLDivLossGrad with batchmean reduction mode.
    Description: KLDivLossGrad with batchmean reduction mode, 2d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = mindspore.Tensor(np.array([-1]).astype(dtype))
    net = Net("batchmean")
    output = net(prediction, target, dy)
    print(output)
    expect = np.array([[0, 0.5], [0.5, 0]])
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float32])
def test_reduction_default(dtype):
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with error default mode, 2d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = mindspore.Tensor(np.array([-1]).astype(dtype))
    net = Net()
    output = net(prediction, target, dy)
    print(output)
    expect = np.array([[0, 0.25], [0.25, 0]]).astype(dtype)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float32])
def test_reduction_error(dtype):
    """
    Feature: KLDivLossGrad with error reduction mode.
    Description: KLDivLossGrad with error reduction mode, 2d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = mindspore.Tensor(np.array([-1]).astype(dtype))
    with pytest.raises(ValueError):
        net = Net("error")
        net(prediction, target, dy)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float16])
def test_reduction_not_str(dtype):
    """
    Feature: KLDivLossGrad with float reduction mode.
    Description: KLDivLossGrad with float reduction mode, 2d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = mindspore.Tensor(np.array([-1]).astype(dtype))
    with pytest.raises(ValueError):
        net = Net(1.0)
        net(prediction, target, dy)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_input_dtype_str():
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with default reduction mode, str input.
    Expectation: run success without error.
    """
    with pytest.raises(TypeError):
        prediction = mindspore.Tensor('0.3')
        target = mindspore.Tensor('1')
        dy = mindspore.Tensor(np.array([-1]).astype(np.float32))
        net = Net()
        net(prediction, target, dy)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float32])
def test_input_0d(dtype):
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with default reduction mode, 0d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array(0.3).astype(dtype))
    target = mindspore.Tensor(np.array(1).astype(dtype))
    dy = mindspore.Tensor(np.array(-1).astype(dtype))
    net = Net("none")
    output = net(prediction, target, dy)
    print(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float32])
def test_input_1d(dtype):
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with default reduction mode, 1d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([0.3]).astype(dtype))
    target = mindspore.Tensor(np.array([1]).astype(dtype))
    dy = mindspore.Tensor(np.array(-1).astype(dtype))
    net = Net("mean")
    output = net(prediction, target, dy)
    print(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float32])
def test_input_3d(dtype):
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with default reduction mode, 3d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[[0.3, 0.7], [0.5, 0.5]]]).astype(dtype))
    target = mindspore.Tensor(np.array([[[-1, 1], [1, -1]]]).astype(dtype))
    dy = mindspore.Tensor(np.array(-1).astype(dtype))
    net = Net("mean")
    output = net(prediction, target, dy)
    print(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float16])
def test_input_4d(dtype):
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with default reduction mode, 4d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[[[0.3, 0.7], [0.5, 0.5]]]]).astype(dtype))
    target = mindspore.Tensor(np.array([[[[-1, 1], [1, -1]]]]).astype(dtype))
    dy = mindspore.Tensor(np.array(-1).astype(dtype))
    net = Net("mean")
    output = net(prediction, target, dy)
    print(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float16])
def test_input_5d(dtype):
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with default reduction mode, 5d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[[[[0.3, 0.7], [0.5, 0.5]]]]]).astype(dtype))
    target = mindspore.Tensor(np.array([[[[[-1, 1], [1, -1]]]]]).astype(dtype))
    dy = mindspore.Tensor(np.array(-1).astype(dtype))
    net = Net("mean")
    output = net(prediction, target, dy)
    print(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float64])
def test_input_6d(dtype):
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with default reduction mode, 6d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[[[[[0.3, 0.7], [0.5, 0.5]]]]]]).astype(dtype))
    target = mindspore.Tensor(np.array([[[[[[-1, 1], [1, -1]]]]]]).astype(dtype))
    dy = mindspore.Tensor(np.array(-1).astype(dtype))
    net = Net("mean")
    output = net(prediction, target, dy)
    print(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [np.float64])
def test_input_7d(dtype):
    """
    Feature: KLDivLossGrad with default reduction mode.
    Description: KLDivLossGrad with default reduction mode, 7d input.
    Expectation: run success without error.
    """
    prediction = mindspore.Tensor(np.array([[[[[[[0.3, 0.7], [0.5, 0.5]]]]]]]).astype(dtype))
    target = mindspore.Tensor(np.array([[[[[[[-1, 1], [1, -1]]]]]]]).astype(dtype))
    dy = mindspore.Tensor(np.array(-1).astype(dtype))
    net = Net("mean")
    output = net(prediction, target, dy)
    print(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap_case():
    """
    Feature: KLDivLossGrad with vmap mode.
    Description: KLDivLossGrad with vmap mode, 2d input.
    Expectation: run success without error.
    """
    class NetVmap(nn.Cell):
        def __init__(self, reduction="none"):
            super(NetVmap, self).__init__()
            if reduction is not None:
                self.kl_div_loss_grad = G.KLDivLossGrad(reduction)
            else:
                self.kl_div_loss_grad = G.KLDivLossGrad()

        def construct(self, dy, x, y):
            return self.kl_div_loss_grad(dy, x, y)

    class WrapNet(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, x, y, dy):
            return vmap(self.net, self.in_axes, self.out_axes)(dy, x, y)

    dtype = np.float32
    prediction = mindspore.Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = mindspore.Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = mindspore.Tensor(np.array([[-1, 0], [1, 1]]).astype(dtype))
    output = WrapNet(NetVmap(), 0, 0)(prediction, target, dy)
    print(output)
    expect = np.array([[0, 0], [-1, 0]])
    assert np.allclose(output.asnumpy(), expect)
