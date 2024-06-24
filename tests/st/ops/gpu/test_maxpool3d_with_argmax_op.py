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

import mindspore.ops.operations as P
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore.ops.functional import vmap
from mindspore.ops.operations import nn_ops as NN

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class MaxPool3DWithArgmaxNet(Cell):
    def __init__(self, ksize, strides, pads, dilation, ceil_mode,
                 data_format="NCDHW", argmax_type=mstype.int64):
        super(MaxPool3DWithArgmaxNet, self).__init__()
        self.maxpool3d_with_argmax = NN.MaxPool3DWithArgmax(
            ksize=ksize, strides=strides, pads=pads, dilation=dilation,
            ceil_mode=ceil_mode, data_format=data_format, argmax_type=argmax_type)

    def construct(self, input_data):
        output, argmax = self.maxpool3d_with_argmax(input_data)
        return output, argmax


class DynamicShapeMaxPool3DWithArgmaxNet(Cell):
    def __init__(self, net, axis=0):
        super(DynamicShapeMaxPool3DWithArgmaxNet, self).__init__()
        self.net = net
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.axis = axis

    def construct(self, x, indices):
        unique_indices, _ = self.unique(indices)
        x = self.gather(x, unique_indices, self.axis)
        return self.net(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool3d_withargmax_float32():
    """
    Feature: Test MaxPool3DWithArgmax.
    Description: Test MaxPool3DWithArgmax with float32 inputs.
    Expectation: success.
    """
    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW', 'argmax_type': mstype.int64}
    inputs = Tensor(np.random.randn(5, 4, 3, 4, 3).astype(np.float32))
    net = MaxPool3DWithArgmaxNet(**attributes)
    output, argmax = net(inputs)
    assert output.shape == (5, 4, 1, 2, 1)
    assert argmax.shape == (5, 4, 1, 2, 1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool3d_withargmax_float16():
    """
    Feature: Test MaxPool3DWithArgmax.
    Description: Test MaxPool3DWithArgmax with float16 inputs.
    Expectation: success.
    """
    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW', 'argmax_type': mstype.int64}
    inputs = Tensor(np.random.randn(5, 4, 3, 4, 3).astype(np.float16))
    net = MaxPool3DWithArgmaxNet(**attributes)
    output, argmax = net(inputs)
    assert output.shape == (5, 4, 1, 2, 1)
    assert argmax.shape == (5, 4, 1, 2, 1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool3d_withargmax_vmap():
    """
    Feature: Test vmap.
    Description: Test MaxPool3DWithArgmax with vmap.
    Expectation: success.
    """
    inputs = Tensor(np.random.randn(5, 4, 3, 4, 3, 2).astype(np.float16))
    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW', 'argmax_type': mstype.int64}
    net = MaxPool3DWithArgmaxNet(**attributes)
    nest_vmap = vmap(net, in_axes=-1, out_axes=0)
    out, indices = nest_vmap(inputs)
    expect_shape = (2, 5, 4, 1, 2, 1)
    assert out.shape == expect_shape
    assert indices.shape == expect_shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_maxpool3d_with_argmax():
    """
    Feature: MaxPool3DWithArgmax dynamic test.
    Description: Run unique and gather ops before MaxPool3DWithArgmax.
    Expectation: success.
    """
    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW', 'argmax_type': mstype.int64}
    inputs = Tensor(np.random.randn(5, 4, 3, 4, 3).astype(np.float16))
    indices = Tensor(np.array([0, 1, 2, 3, 0]).astype(np.int32))
    net = MaxPool3DWithArgmaxNet(**attributes)
    dy_net = DynamicShapeMaxPool3DWithArgmaxNet(net)
    output, argmax = dy_net(inputs, indices)
    assert output.shape == (4, 4, 1, 2, 1)
    assert argmax.shape == (4, 4, 1, 2, 1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool3d_with_argmax_dynamic_shape():
    """
    Feature: MaxPool3DWithArgmax dynamic test.
    Description: test MaxPool3DWithArgmax with dynamic shape.
    Expectation: success.
    """

    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW', 'argmax_type': mstype.int64}
    x = Tensor(np.random.randn(5, 4, 3, 4, 3).astype(np.float32))
    x_dyn = Tensor(shape=[None for _ in x.shape], dtype=mstype.float32)
    net = MaxPool3DWithArgmaxNet(**attributes)
    net.set_inputs(x_dyn)
    output, argmax = net(x)
    assert output.shape == (5, 4, 1, 2, 1)
    assert argmax.shape == (5, 4, 1, 2, 1)
