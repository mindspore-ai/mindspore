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
from mindspore import Tensor, Parameter
from mindspore.ops.functional import vmap


def vmap_case():
    class Net(nn.Cell):
        def __init__(self, axis):
            super(Net, self).__init__()
            self.index_add = ops.IndexAdd(axis)

        def construct(self, a, idx, b):
            return self.index_add(a, idx, b)

    class WrapNet(nn.Cell):
        def __init__(self, net, a, in_axes, out_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.a = a
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, idx, b):
            return vmap(self.net, self.in_axes, self.out_axes)(self.a, idx, b)

    # batch dimension of x and y is same, batch dimension <= axis
    x = Parameter(Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)))
    indices = Tensor(np.array([0, 2], dtype=np.int32))
    y = Tensor(np.array([[0.5, 1], [1, 1.5], [2, 2.5]], dtype=np.float32))
    output = WrapNet(Net(0), x, (0, None, 0), 0)(indices, y)
    expect = np.array([[1.5, 2, 4], [5, 5, 7.5], [9, 8, 11.5]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect)

    # batch dimension of x and y is different, batch dimension <= axis
    x = Parameter(Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)))
    indices = Tensor(np.array([0, 2], dtype=np.int32))
    y = Tensor(np.array([[0.5, 1, 2], [1, 1.5, 2.5]], dtype=np.float32))
    output = WrapNet(Net(0), x, (0, None, 1), 0)(indices, y)
    expect = np.array([[1.5, 2, 4], [5, 5, 7.5], [9, 8, 11.5]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect)

    # batch dimension y is None
    x = Parameter(Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)))
    indices = Tensor(np.array([0, 2], dtype=np.int32))
    y = Tensor(np.array([0.5, 1], dtype=np.float32))
    output = WrapNet(Net(0), x, (0, None, None), 0)(indices, y)
    expect = np.array([[1.5, 2, 4], [4.5, 5, 7], [7.5, 8, 10]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect)

    # batch dimension of x and y is same, batch dimension > axis
    x = Parameter(Tensor(np.array([[[1, 1], [1, 1]],
                                   [[2, 2], [2, 2]],
                                   [[3, 3], [3, 3]]], dtype=np.float32)))
    indices = Tensor(np.array([0, 2], dtype=np.int32))
    y = Tensor(np.array([[[0, 0.5], [1, 1.5]], [[1.5, 2], [2.5, 3]]], dtype=np.float32))
    output = WrapNet(Net(0), x, (2, None, 2), 2)(indices, y)
    expect = np.array([[[1, 1.5], [2, 2.5]],
                       [[2, 2], [2, 2]],
                       [[4.5, 5], [5.5, 6]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_index_add_vmap_cpu():
    """
    Feature: test IndexAdd vmap on CPU.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    vmap_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_vmap_gpu():
    """
    Feature: test IndexAdd vmap on GPU.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    vmap_case()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_index_add_vmap_ascend():
    """
    Feature: test IndexAdd vmap on Ascend.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    vmap_case()
