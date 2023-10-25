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

import numpy as np
import pytest

import mindspore
import mindspore.nn as nn
import mindspore.context as context
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_im2col_vmap():
    """
    Feature: Test Im2Col CPU vmap
    Description: Test vmap for im2col
    Expectation: Consistent with the assertion
    """
    def cal_im2col(x):
        return P.Im2Col(ksizes=3,
                        strides=1,
                        dilations=1,
                        pads=0)(x)

    # once vmap
    x1 = Tensor(
        np.arange(4 * 4 * 32 * 32).reshape(1, 4, 4, 32, 32), mindspore.float64)
    vmap_im2col = vmap(cal_im2col, in_axes=-1)
    outputs = vmap_im2col(x1)
    assert outputs.asnumpy().shape == (32, 1, 4, 9, 60)

    # twice vmap
    x2 = Tensor(
        np.arange(4 * 4 * 32 * 32).reshape(1, 1, 4, 4, 32, 32),
        mindspore.float64)
    vmap_im2col = vmap(vmap(cal_im2col, in_axes=-1), in_axes=-1)
    outputs = vmap_im2col(x2)
    assert outputs.asnumpy().shape == (32, 32, 1, 1, 9, 4)


class NetIm2Col(nn.Cell):
    def __init__(self):
        super(NetIm2Col, self).__init__()
        self.im2col = P.Im2Col(ksizes=2,
                               strides=1,
                               dilations=1,
                               pads=0)

    def construct(self, x):
        return self.im2col(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_im2col_cpu_dynamic_shape():
    """
    Feature: test Im2Col op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = NetIm2Col()

    x_dyn = Tensor(shape=[None, 32, 9, 9], dtype=mindspore.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 4, 64)
    assert output.asnumpy().shape == expect_shape

    x_dyn = Tensor(shape=[1, None, 9, 9], dtype=mindspore.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 4, 64)
    assert output.asnumpy().shape == expect_shape

    x_dyn = Tensor(shape=[1, 32, None, None], dtype=mindspore.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 4, 64)
    assert output.asnumpy().shape == expect_shape
