# Copyright 2020 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class NetGatherV2_axis0(nn.Cell):
    def __init__(self):
        super(NetGatherV2_axis0, self).__init__()
        self.gatherv2 = P.Gather()

    def construct(self, params, indices):
        return self.gatherv2(params, indices, 0)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherv2_axis0():
    x = Tensor(np.arange(3 * 2 * 2).reshape(3, 2, 2), mstype.float32)
    indices = Tensor(np.array([1, 2]), mstype.int32)
    gatherv2 = NetGatherV2_axis0()
    ms_output = gatherv2(x, indices)
    print("output:\n", ms_output)
    expect = np.array([[[4., 5.],
                        [6., 7.]],
                       [[8., 9.],
                        [10., 11.]]])
    error = np.ones(shape=ms_output.asnumpy().shape) * 1.0e-6
    diff = ms_output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

class NetGatherV2_axis1(nn.Cell):
    def __init__(self):
        super(NetGatherV2_axis1, self).__init__()
        self.gatherv2 = P.Gather()

    def construct(self, params, indices):
        return self.gatherv2(params, indices, 1)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherv2_axis1():
    x = Tensor(np.arange(2 * 3 * 2).reshape(2, 3, 2), mstype.float32)
    indices = Tensor(np.array([1, 2]), mstype.int32)
    gatherv2 = NetGatherV2_axis1()
    ms_output = gatherv2(x, indices)
    print("output:\n", ms_output)
    expect = np.array([[[2., 3.],
                        [4., 5.]],
                       [[8., 9.],
                        [10., 11.]]])
    error = np.ones(shape=ms_output.asnumpy().shape) * 1.0e-6
    diff = ms_output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

class NetGatherV2_axisN1(nn.Cell):
    def __init__(self):
        super(NetGatherV2_axisN1, self).__init__()
        self.gatherv2 = P.Gather()

    def construct(self, params, indices):
        return self.gatherv2(params, indices, -1)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherv2_axisN1():
    x = Tensor(np.arange(2 * 2 * 3).reshape(2, 2, 3), mstype.float32)
    indices = Tensor(np.array([1, 2]), mstype.int32)
    gatherv2 = NetGatherV2_axisN1()
    ms_output = gatherv2(x, indices)
    print("output:\n", ms_output)
    expect = np.array([[[1., 2.],
                        [4., 5.]],
                       [[7., 8.],
                        [10., 11.]]])
    error = np.ones(shape=ms_output.asnumpy().shape) * 1.0e-6
    diff = ms_output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

if __name__ == '__main__':
    test_gatherv2_axis0()
    test_gatherv2_axis1()
    test_gatherv2_axisN1()
