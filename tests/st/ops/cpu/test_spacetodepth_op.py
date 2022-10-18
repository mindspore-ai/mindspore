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
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
class SpaceToDepthNet(nn.Cell):
    def __init__(self, nptype):
        super(SpaceToDepthNet, self).__init__()
        self.SpaceToDepth = P.SpaceToDepth(2)

        data_np = np.array([[[[0, 3],
                              [6, 9]],
                             [[1, 4],
                              [7, 10]],
                             [[2, 5],
                              [8, 11]]]]).astype(nptype)
        self.data_np = data_np
        self.x = Parameter(initializer(Tensor(self.data_np), (1, 3, 2, 2)), name='x')

    @jit
    def construct(self):
        return self.SpaceToDepth(self.x)


def SpaceToDepth(nptype):
    expect = np.arange(12).reshape((1, 12, 1, 1)).astype(nptype)
    std = SpaceToDepthNet(nptype)
    output = std()
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_float32():
    SpaceToDepth(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_float16():
    SpaceToDepth(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_int32():
    SpaceToDepth(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_int64():
    SpaceToDepth(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_int8():
    SpaceToDepth(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_int16():
    SpaceToDepth(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint8():
    SpaceToDepth(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint16():
    SpaceToDepth(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint32():
    SpaceToDepth(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint64():
    SpaceToDepth(np.uint64)
