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

import mindspore.ops.operations as P
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.ops.functional import vmap
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class MaxPool3DGradWithArgmaxNet(Cell):
    def __init__(self, ksize, strides, pads, dilation, ceil_mode=False,
                 data_format="NCDHW"):
        super(MaxPool3DGradWithArgmaxNet, self).__init__()
        self.maxpool3d_grad_with_argmax = G.MaxPool3DGradWithArgmax(
            ksize=ksize, strides=strides, pads=pads, dilation=dilation,
            ceil_mode=ceil_mode, data_format=data_format)

    def construct(self, x, dy, mask):
        output = self.maxpool3d_grad_with_argmax(x, dy, mask)
        return output


class DynamicShapeMaxPool3DGradWithArgmaxNet(Cell):
    def __init__(self, net, axis=0):
        super(DynamicShapeMaxPool3DGradWithArgmaxNet, self).__init__()
        self.net = net
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.axis = axis

    def construct(self, x, dy, mask, indices):
        unique_indices, _ = self.unique(indices)
        x = self.gather(x, unique_indices, self.axis)
        dy = self.gather(dy, unique_indices, self.axis)
        mask = self.gather(mask, unique_indices, self.axis)
        return self.net(x, dy, mask)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool3d_grad_withargmax_float32():
    """
    Feature: Test MaxPool3DGradWithArgmax.
    Description: Test MaxPool3DGradWithArgmax with float32 inputs.
    Expectation: success.
    """
    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW'}
    inputs = Tensor(np.arange(3*4*3).reshape(1, 1, 3, 4, 3).astype(np.float32))
    dy = Tensor(np.ones((1, 1, 1, 2, 1)).astype(np.float32))
    mask = Tensor(np.array([[[[[32], [35]]]]]).astype(np.int32))
    net = MaxPool3DGradWithArgmaxNet(**attributes)
    output = net(inputs, dy, mask)
    expect = np.array([[[[[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.]],
                         [[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.]],
                         [[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 1.],
                          [0., 0., 1.]]]]]).astype(np.float32)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool3d_grad_withargmax_float16():
    """
    Feature: Test MaxPool3DGradWithArgmax.
    Description: Test MaxPool3DGradWithArgmax with float16 inputs.
    Expectation: success.
    """
    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW'}
    inputs = Tensor(np.arange(3*4*3).reshape(1, 1, 3, 4, 3).astype(np.float16))
    dy = Tensor(np.ones((1, 1, 1, 2, 1)).astype(np.float16))
    mask = Tensor(np.array([[[[[32], [35]]]]]).astype(np.int32))
    net = MaxPool3DGradWithArgmaxNet(**attributes)
    output = net(inputs, dy, mask)
    expect = np.array([[[[[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.]],
                         [[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.]],
                         [[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 1.],
                          [0., 0., 1.]]]]]).astype(np.float16)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool3d_grad_withargmax_vmap():
    """
    Feature: Test vmap.
    Description: Test MaxPool3DGradWithArgmax with vmap.
    Expectation: success.
    """
    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW'}
    net = MaxPool3DGradWithArgmaxNet(**attributes)
    nest_vmap = vmap(net, in_axes=(-1, -1, -1), out_axes=0)
    inputs = Tensor(np.arange(3*4*3).reshape(1, 1, 3, 4, 3, 1).astype(np.float32))
    dy = Tensor(np.ones((1, 1, 1, 2, 1, 1)).astype(np.float32))
    mask = Tensor(np.array([[[[[[32]], [[35]]]]]]).astype(np.int32))
    out = nest_vmap(inputs, dy, mask)
    assert out.shape == (1, 1, 1, 3, 4, 3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_maxpool3d_grad_with_argmax():
    """
    Feature: MaxPool3DGradWithArgmax dynamic test.
    Description: Run unique and gather ops before MaxPool3DGradWithArgmax.
    Expectation: success.
    """
    attributes = {'ksize': 3, 'strides': 1, 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'data_format': 'NCDHW'}
    inputs = Tensor(np.arange(3*4*3).reshape(1, 1, 3, 4, 3).astype(np.float32))
    dy = Tensor(np.ones((1, 1, 1, 2, 1)).astype(np.float32))
    mask = Tensor(np.array([[[[[32], [35]]]]]).astype(np.int32))
    indices = Tensor(np.array([0]).astype(np.int32))
    net = MaxPool3DGradWithArgmaxNet(**attributes)
    dy_net = DynamicShapeMaxPool3DGradWithArgmaxNet(net)
    out = dy_net(inputs, dy, mask, indices)
    assert out.shape == inputs.shape
