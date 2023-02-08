# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context

class GatherNdNet(nn.Cell):
    def __init__(self):
        super(GatherNdNet, self).__init__()
        self.gathernd = P.GatherNd()

    def construct(self, x, indices):
        return self.gathernd(x, indices)


def gathernd0(nptype):
    x = Tensor(np.arange(3 * 2, dtype=nptype).reshape(3, 2))
    indices = Tensor(np.array([[1, 1], [0, 1]]).astype(np.int32))
    expect = np.array([3, 1]).astype(nptype)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gathernd = GatherNdNet()
    output = gathernd(x, indices)

    assert np.array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd0_float64():
    gathernd0(np.float64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd0_float32():
    gathernd0(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd0_float16():
    gathernd0(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd0_int32():
    gathernd0(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd0_int16():
    gathernd0(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd0_uint8():
    gathernd0(np.uint8)

def gathernd1(nptype):
    x = Tensor(np.arange(2 * 3 * 4 * 5, dtype=nptype).reshape(2, 3, 4, 5))
    indices = Tensor(np.array([[[[[l, k, j, i] for i in [1, 3, 4]] for j in range(4)]
                                for k in range(3)] for l in range(2)], dtype='i4'))
    expect = np.array([[[[1., 3., 4.],
                         [6., 8., 9.],
                         [11., 13., 14.],
                         [16., 18., 19.]],

                        [[21., 23., 24.],
                         [26., 28., 29.],
                         [31., 33., 34.],
                         [36., 38., 39.]],

                        [[41., 43., 44.],
                         [46., 48., 49.],
                         [51., 53., 54.],
                         [56., 58., 59.]]],

                       [[[61., 63., 64.],
                         [66., 68., 69.],
                         [71., 73., 74.],
                         [76., 78., 79.]],

                        [[81., 83., 84.],
                         [86., 88., 89.],
                         [91., 93., 94.],
                         [96., 98., 99.]],

                        [[101., 103., 104.],
                         [106., 108., 109.],
                         [111., 113., 114.],
                         [116., 118., 119.]]]]).astype(nptype)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gather = GatherNdNet()
    output = gather(x, indices)

    assert np.array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd1_float64():
    gathernd1(np.float64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd1_float32():
    gathernd1(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd1_float16():
    gathernd1(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd1_int32():
    gathernd1(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd1_int16():
    gathernd1(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd1_uint8():
    gathernd1(np.uint8)

def gathernd2(nptype):
    x = Tensor(np.array([[4., 5., 4., 1., 5.],
                         [4., 9., 5., 6., 4.],
                         [9., 8., 4., 3., 6.],
                         [0., 4., 2., 2., 8.],
                         [1., 8., 6., 2., 8.],
                         [8., 1., 9., 7., 3.],
                         [7., 9., 2., 5., 7.],
                         [9., 8., 6., 8., 5.],
                         [3., 7., 2., 7., 4.],
                         [4., 2., 8., 2., 9.]]).astype(np.float16))

    indices = Tensor(np.array([[0], [1], [3]]).astype(np.int32))
    expect = np.array([[4., 5., 4., 1., 5.],
                       [4., 9., 5., 6., 4.],
                       [0., 4., 2., 2., 8.]])

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gathernd = GatherNdNet()
    output = gathernd(x, indices)

    assert np.array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd2_float64():
    gathernd2(np.float64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd2_float32():
    gathernd2(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd2_float16():
    gathernd2(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd2_int32():
    gathernd2(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd2_int16():
    gathernd2(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd2_uint8():
    gathernd2(np.uint8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd_bool():
    x = Tensor(np.array([[True, False], [False, False]]).astype(np.bool))
    indices = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.int32))
    expect = np.array([True, False, False, False]).astype(np.bool)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gathernd = GatherNdNet()
    output = gathernd(x, indices)

    assert np.array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gathernd_indices_int64():
    x = Tensor(np.array([[True, False], [False, False]]).astype(np.bool))
    indices = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.int64))
    expect = np.array([True, False, False, False]).astype(np.bool)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gathernd = GatherNdNet()
    output = gathernd(x, indices)

    assert np.array_equal(output.asnumpy(), expect)
