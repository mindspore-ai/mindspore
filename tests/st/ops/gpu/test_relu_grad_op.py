# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _grad_ops as G


class NetReluGrad(nn.Cell):
    def __init__(self):
        super(NetReluGrad, self).__init__()
        self.rekuGrad = G.ReluGrad()

    def construct(self, x, dy):
        return self.rekuGrad(dy, x)


def relu_grad_base(dtype):
    x = Tensor(np.array([[[[-1, 1, 1],
                           [1, -1, 1],
                           [1, 1, -1]]]]).astype(dtype))
    dy = Tensor(np.array([[[[1, 0, 1],
                            [0, 1, 0],
                            [1, 1, 1]]]]).astype(dtype))
    expect = np.array([[[[0, 0, 1,], [0, 0, 0,], [1, 1, 0.]]]]).astype(np.dtype)
    error = np.ones(shape=[3, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu_grad = NetReluGrad()
    output = relu_grad(x, dy)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert output.asnumpy().dtype == dtype


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_grad_float16():
    relu_grad_base(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_grad_float32():
    relu_grad_base(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_grad_int8():
    relu_grad_base(np.int8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_grad_int32():
    relu_grad_base(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_grad_int64():
    relu_grad_base(np.int64)
