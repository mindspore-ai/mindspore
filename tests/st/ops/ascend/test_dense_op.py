# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import ParameterTuple
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations import nn_ops as ops


class Dense(nn.Cell):
    def __init__(self):
        super(Dense, self).__init__()
        self.dense = ops.Dense()

    def construct(self, x, w, b):
        return self.dense(x, w, b)


class DenseGrad(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.grad = GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        return self.grad(self.network, self.params)(*inputs)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_1d_forward():
    """
    Feature: Test dense 1d.
    Description: Test dense 1d forward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dtype = np.float32
    error = 1e-3
    x_np = np.array([1, 2, 3]).astype(dtype)
    w_np = np.array([1, 2, 3]).astype(dtype)
    b_np = np.array(7).astype(dtype)
    out_np = np.array(21).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    net = Dense()
    net.set_train()

    # dynamic shape
    net.set_inputs(
        Tensor(shape=[None for _ in x_ms.shape], dtype=x_ms.dtype),
        Tensor(shape=[None for _ in w_ms.shape], dtype=w_ms.dtype),
        b_ms,
    )
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    assert np.abs(out_ms - out_np).mean() < error

    # dynamic rank
    net.set_inputs(
        Tensor(dtype=x_ms.dtype),
        Tensor(dtype=w_ms.dtype),
        b_ms,
    )
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    assert np.abs(out_ms - out_np).mean() < error


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3d_forward():
    """
    Feature: Test dense 3d forward.
    Description: Test dense 3d forward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dtype = np.float32
    error = 1e-3
    x_np = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(dtype)
    w_np = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]).astype(dtype)
    b_np = np.array([19, 20, 21, 22]).astype(dtype)
    out_np = np.array([[69, 88, 107, 126], [141, 187, 233, 279]]).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    net = Dense()
    net.set_train()

    # dynamic shape
    net.set_inputs(
        Tensor(shape=[None for _ in x_ms.shape], dtype=x_ms.dtype),
        Tensor(shape=[None for _ in w_ms.shape], dtype=w_ms.dtype),
        b_ms,
    )
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    assert np.abs(out_ms - out_np).mean() < error

    # dynamic rank
    net.set_inputs(
        Tensor(dtype=x_ms.dtype),
        Tensor(dtype=w_ms.dtype),
        b_ms,
    )
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    assert np.abs(out_ms - out_np).mean() < error


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_1d_backward():
    """
    Feature: Test dense 1d backward.
    Description: Test dense 1d backward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dtype = np.float32
    error = 1e-3
    x_np = np.array([1, 2, 3]).astype(dtype)
    w_np = np.array([4, 5, 6]).astype(dtype)
    b_np = np.array(7).astype(dtype)
    dout_np = np.array(8).astype(dtype)
    dx_np = np.array([32, 40, 48]).astype(dtype)
    dw_np = np.array([8, 16, 24]).astype(dtype)
    db_np = np.array(8).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    dout_ms = Tensor(dout_np)
    net = Dense()
    grad_net = DenseGrad(net)
    grad_net.set_train()

    # dynamic shape
    grad_net.set_inputs(
        Tensor(dtype=x_ms.dtype),
        Tensor(dtype=w_ms.dtype),
        b_ms,
        dout_ms,
    )
    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error

    # dynamic rank
    grad_net.set_inputs(
        Tensor(shape=[None for _ in x_ms.shape], dtype=x_ms.dtype),
        Tensor(shape=[None for _ in w_ms.shape], dtype=w_ms.dtype),
        b_ms,
        dout_ms,
    )
    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3d_backward():
    """
    Feature: Test dense 3d backward.
    Description: Test dense 3d backward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dtype = np.float32
    error = 1e-3
    x_np = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(dtype)
    w_np = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]).astype(dtype)
    b_np = np.array([19, 20, 21, 22]).astype(dtype)
    dout_np = np.array([[[23, 24, 25, 26], [27, 28, 29, 30]]]).astype(np.float32)
    dx_np = np.array([[[1142, 1240, 1338], [1326, 1440, 1554]]]).astype(dtype)
    dw_np = np.array([[131, 181, 231], [136, 188, 240], [141, 195, 249.], [146, 202, 258]]).astype(dtype)
    db_np = np.array([50, 52, 54, 56]).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    dout_ms = Tensor(dout_np)
    net = Dense()
    grad_net = DenseGrad(net)
    grad_net.set_train()

    # dynamic shape
    grad_net.set_inputs(
        Tensor(dtype=x_ms.dtype),
        Tensor(dtype=w_ms.dtype),
        b_ms,
        dout_ms,
    )
    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error

    # dynamic rank
    grad_net.set_inputs(
        Tensor(shape=[None for _ in x_ms.shape], dtype=x_ms.dtype),
        Tensor(shape=[None for _ in w_ms.shape], dtype=w_ms.dtype),
        b_ms,
        dout_ms,
    )
    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error
