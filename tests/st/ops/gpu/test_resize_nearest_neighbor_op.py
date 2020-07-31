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
from mindspore.ops import operations as P


class ResizeNearestNeighborAlignCornerT(nn.Cell):
    def __init__(self, size):
        super(ResizeNearestNeighborAlignCornerT, self).__init__()
        self.ResizeNearestNeighborAlignCornerT = P.ResizeNearestNeighbor(size, align_corners=True)

    def construct(self, x):
        return self.ResizeNearestNeighborAlignCornerT(x)

class ResizeNearestNeighborAlignCornerF(nn.Cell):
    def __init__(self, size):
        super(ResizeNearestNeighborAlignCornerF, self).__init__()
        self.ResizeNearestNeighborAlignCornerF = P.ResizeNearestNeighbor(size, align_corners=False)

    def construct(self, x):
        return self.ResizeNearestNeighborAlignCornerF(x)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ResizeNearestNeighborAlignCornerT():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    input_tensor = Tensor(np.array([[[[1, 0], [0, 1]]]]).astype(np.float32))
    expect = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float32)
    rnn = ResizeNearestNeighborAlignCornerT((4, 4))
    output = rnn(input_tensor)
    assert np.all(output.asnumpy() == expect)
    input_tensor = Tensor(np.array([[[[1, 0], [0, 1]]]]).astype(np.float16))
    expect = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float16)
    rnn = ResizeNearestNeighborAlignCornerT((4, 4))
    output = rnn(input_tensor)
    assert np.all(output.asnumpy() == expect)
    input_tensor = Tensor(np.array([[[[1, 0], [0, 1]]]]).astype(np.int32))
    expect = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.int32)
    rnn = ResizeNearestNeighborAlignCornerT((4, 4))
    output = rnn(input_tensor)
    assert np.all(output.asnumpy() == expect)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ResizeNearestNeighborAlignCornerF():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    input_tensor = Tensor(np.array([[[[1, 0], [0, 1]]]]).astype(np.float32))
    expect = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float32)
    rnn = ResizeNearestNeighborAlignCornerF((4, 4))
    output = rnn(input_tensor)
    assert np.all(output.asnumpy() == expect)
    input_tensor = Tensor(np.array([[[[1, 0], [0, 1]]]]).astype(np.float16))
    expect = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float16)
    rnn = ResizeNearestNeighborAlignCornerF((4, 4))
    output = rnn(input_tensor)
    assert np.all(output.asnumpy() == expect)
    input_tensor = Tensor(np.array([[[[1, 0], [0, 1]]]]).astype(np.int32))
    expect = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.int32)
    rnn = ResizeNearestNeighborAlignCornerF((4, 4))
    output = rnn(input_tensor)
    assert np.all(output.asnumpy() == expect)
