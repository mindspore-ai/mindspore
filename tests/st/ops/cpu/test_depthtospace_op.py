# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
class DepthToSpaceNet(nn.Cell):
    def __init__(self, nptype, block_size=2, input_shape=(1, 12, 1, 1)):
        super(DepthToSpaceNet, self).__init__()
        self.DepthToSpace = P.DepthToSpace(block_size)

        input_size = 1
        for i in input_shape:
            input_size = input_size*i
        data_np = np.arange(input_size).reshape(input_shape).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(data_np), input_shape), name='x1')

    @ms_function
    def construct(self):
        y1 = self.DepthToSpace(self.x1)
        return y1


def DepthToSpace(nptype, block_size=2, input_shape=(1, 12, 1, 1)):
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.array([[[[0, 3],
                         [6, 9]],
                        [[1, 4],
                         [7, 10]],
                        [[2, 5],
                         [8, 11]]]]).astype(nptype)
    dts = DepthToSpaceNet(nptype, block_size, input_shape)
    output = dts()
    assert (output.asnumpy() == expect).all()


class DynamicShapeNet(nn.Cell):
    def __init__(self, block_size):
        super(DynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.depth_to_space = P.DepthToSpace(block_size)

    def construct(self, x, indices):
        unique_indices, _ = self.unique(indices)
        x = self.gather(x, unique_indices, 0)
        return self.depth_to_space(x)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depth_to_space_dynamic_shape():
    """
    Feature: test DepthToSpace dynamic_shape feature.
    Description: test DepthToSpace dynamic_shape feature.
    Expectation: success.
    """
    # dynamic inputs
    indices_np = np.random.randint(0, 3, size=6)
    indices_ms = Tensor(indices_np)

    # data preparation
    x = Tensor(np.random.rand(4, 12, 1, 1), mstype.float32)

    # dynamic shape
    x_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    dynamic_shape_net = DynamicShapeNet(2)
    dynamic_shape_net.set_inputs(x_dyn, indices_ms)

    # run in graph mode
    outputs = dynamic_shape_net(x, indices_ms)
    expect_shape = (len(np.unique(indices_np)), 3, 2, 2)
    assert outputs.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_float32():
    DepthToSpace(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_float16():
    DepthToSpace(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_int32():
    DepthToSpace(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_int64():
    DepthToSpace(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_int8():
    DepthToSpace(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_int16():
    DepthToSpace(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_uint8():
    DepthToSpace(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_uint16():
    DepthToSpace(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_uint32():
    DepthToSpace(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_depthtospace_graph_uint64():
    DepthToSpace(np.uint64)
