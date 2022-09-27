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
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations.nn_ops import AdaptiveMaxPool3D
from mindspore.ops.functional import vmap
import mindspore.numpy as ms_np


context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.adaptive_max_pool3d = AdaptiveMaxPool3D()

    def construct(self, x, output_size):
        return self.adaptive_max_pool3d(x, output_size)


class DynamicShapeNet(nn.Cell):
    def __init__(self, axis=0):
        super(DynamicShapeNet, self).__init__()
        self.net = AdaptiveMaxPool3D()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.axis = axis

    def construct(self, x, output_size, indices):
        unique_indices, _ = self.unique(indices)
        x = self.gather(x, unique_indices, self.axis)
        return self.net(x, output_size)


class RankDynamicNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.adaptive_max_pool3d = AdaptiveMaxPool3D()
        self.reduce = P.ReduceSum(keep_dims=False)

    def construct(self, x, output_size):
        rand_axis = ms_np.randint(1, 3, (2,))
        axis = ms_np.unique(rand_axis)
        in_x = self.reduce(x, axis)
        out = self.adaptive_max_pool3d(in_x, output_size)
        return out, in_x


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_rank_dynamic():
    """
    Feature: test rank dynamic shape.
    Description: test AdaptiveMaxPool3D op rank dynamic shape.
    Expectation: expect correct result.
    """
    input_x = Tensor(np.random.randn(8, 3, 5, 8, 5, 6).astype(np.float32))
    output_size = Tensor(np.array([2, 2, 2]).astype(np.int32))
    dyn_net = RankDynamicNet()
    dyn_output, in_x = dyn_net(input_x, output_size)

    net = Net()
    output = net(in_x, output_size)
    assert (dyn_output[0].asnumpy() == output[0].asnumpy()).all()
    assert (dyn_output[1].asnumpy() == output[1].asnumpy()).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_4d():
    """
    Feature: test 4d input shape.
    Description: test AdaptiveMaxPool3D op 5d input shape.
    Expectation: expect correct result.
    """
    x = Tensor(np.random.randn(6, 4, 9, 9).astype(np.float32))
    output_size = Tensor([2, 3, 5], dtype=mstype.int32)
    net = Net()
    output = net(x, output_size)
    expect_shape = (6, 2, 3, 5)
    assert output[0].asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_5d():
    """
    Feature: test 5d input shape.
    Description: test AdaptiveMaxPool3D op 5d input shape.
    Expectation: expect correct result.
    """
    x = Tensor(np.random.randn(2, 6, 4, 9, 9).astype(np.float32))
    output_size = Tensor([2, 3, 5], dtype=mstype.int32)
    net = Net()
    output = net(x, output_size)
    expect_shape = (2, 6, 2, 3, 5)
    assert output[0].asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_dynamic_shape():
    """
    Feature: test dyname shape.
    Description: test AdaptiveMaxPool3D op dynamic shape.
    Expectation: expect correct result.
    """
    x = Tensor(np.random.randn(4, 4, 9, 9).astype(np.float32))
    output_size = Tensor([2, 3, 5], dtype=mstype.int32)
    indices = Tensor([0, 1, 2, 1], dtype=mstype.int32)
    net = DynamicShapeNet()
    output = net(x, output_size, indices)
    expect_shape = (3, 2, 3, 5)
    assert output[0].asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_vmap():
    """
    Feature: test vmap function.
    Description: test AdaptiveMaxPool3D op vmap.
    Expectation: expect correct result.
    """
    in_axes = (-1, None)
    x = Tensor(np.random.randn(4, 4, 9, 9, 2).astype(np.float32))
    output_size = Tensor([2, 3, 5], dtype=mstype.int32)
    net = Net()
    nest_vmap = vmap(net, in_axes=in_axes, out_axes=0)
    out = nest_vmap(x, output_size)
    expect_shape = (2, 4, 2, 3, 5)
    assert out[0].shape == expect_shape
