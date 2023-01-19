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

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class ResizeBilinearGradNet(nn.Cell):
    def __init__(self, align_corners=False):
        super(ResizeBilinearGradNet, self).__init__()
        self.rb1 = G.ResizeBilinearGrad(align_corners=align_corners)

    def construct(self, dy, size, indices_dy, indices_size, axis):
        unique_dy_index, _ = ops.unique(indices_dy)
        unique_size_index, _ = ops.unique(indices_size)
        dy_ = ops.gather(dy, unique_dy_index, axis)
        size_ = ops.gather(size, unique_size_index, axis)
        return self.rb1(dy_, size_)


def dyn_case():
    dy = np.array([[[[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]]]]).astype(np.float32)
    x = np.array([[[[1.1, 2.2], [3.3, 4.4]]]]).astype(np.float32)
    expect = np.array([[[[2.25, 0.75],
                         [0.75, 4.25]]]]).astype(np.float32)
    net = ResizeBilinearGradNet()
    axis = 3
    indices_dy = np.array([i for i in range(dy.shape[axis])])
    indices_x = np.array([i for i in range(x.shape[axis])])
    output = net(Tensor(dy), Tensor(x), Tensor(indices_dy), Tensor(indices_x), axis)
    assert np.all(output.asnumpy() == expect)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resize_bilinear_grad_dyn_ascend():
    """
    Feature: Test ResizeBilinearGrad on Ascend.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dyn_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_bilinear_grad_dyn_gpu():
    """
    Feature: Test ResizeBilinearGrad on GPU.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dyn_case()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_resize_bilinear_grad_dyn_cpu():
    """
    Feature: Test ResizeBilinearGrad on CPU.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dyn_case()
