# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G

class ResizeNearestNeighborGradAlignCornerT(nn.Cell):
    def __init__(self):
        super(ResizeNearestNeighborGradAlignCornerT, self).__init__()
        self.ResizeNearestNeighborGradAlignCornerT = G.ResizeNearestNeighborGrad(align_corners=True)

    def construct(self, dy, size):
        return self.ResizeNearestNeighborGradAlignCornerT(dy, size)

class ResizeNearestNeighborGradAlignCornerF(nn.Cell):
    def __init__(self):
        super(ResizeNearestNeighborGradAlignCornerF, self).__init__()
        self.ResizeNearestNeighborGradAlignCornerF = G.ResizeNearestNeighborGrad(align_corners=False)

    def construct(self, dy, size):
        return self.ResizeNearestNeighborGradAlignCornerF(dy, size)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ResizeNearestNeighborGradAlignCornerT():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float32)
    size = (4, 4)
    expect = np.array([[[[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]]]]).astype(np.float32)
    rnn = ResizeNearestNeighborGradAlignCornerT()
    output = rnn(Tensor(dy), size)
    assert np.all(output.asnumpy() == expect)
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float16)
    size = (4, 4)
    expect = np.array([[[[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]]]]).astype(np.float16)
    rnn = ResizeNearestNeighborGradAlignCornerT()
    output = rnn(Tensor(dy), size)
    assert np.all(output.asnumpy() == expect)
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.int32)
    size = (4, 4)
    expect = np.array([[[[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]]]]).astype(np.int32)
    rnn = ResizeNearestNeighborGradAlignCornerT()
    output = rnn(Tensor(dy), size)
    assert np.all(output.asnumpy() == expect)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ResizeNearestNeighborGradAlignCornerF():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dy = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float32)
    size = (2, 2)
    expect = np.array([[[[4, 0], [0, 4]]]]).astype(np.float32)
    rnn = ResizeNearestNeighborGradAlignCornerF()
    output = rnn(Tensor(dy), size)
    assert np.all(output.asnumpy() == expect)
    dy = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float16)
    size = (2, 2)
    expect = np.array([[[[4, 0], [0, 4]]]]).astype(np.float16)
    rnn = ResizeNearestNeighborGradAlignCornerF()
    output = rnn(Tensor(dy), size)
    assert np.all(output.asnumpy() == expect)
    dy = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.int32)
    size = (2, 2)
    expect = np.array([[[[4, 0], [0, 4]]]]).astype(np.int32)
    rnn = ResizeNearestNeighborGradAlignCornerF()
    output = rnn(Tensor(dy), size)
    assert np.all(output.asnumpy() == expect)
