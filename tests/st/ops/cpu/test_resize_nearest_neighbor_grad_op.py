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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class ResizeNearestNeighborGradAlignCornerT(nn.Cell):
    def __init__(self, size=None):
        super(ResizeNearestNeighborGradAlignCornerT, self).__init__()
        self.ResizeNearestNeighborGradAlignCornerT = G.ResizeNearestNeighborGrad(
            align_corners=True)
        self.size = size

    def construct(self, dy):
        return self.ResizeNearestNeighborGradAlignCornerT(dy, self.size)


class ResizeNearestNeighborGradAlignCornerF(nn.Cell):
    def __init__(self, size=None):
        super(ResizeNearestNeighborGradAlignCornerF, self).__init__()
        self.ResizeNearestNeighborGradAlignCornerF = G.ResizeNearestNeighborGrad(
            align_corners=False)
        self.size = size

    def construct(self, dy):
        return self.ResizeNearestNeighborGradAlignCornerF(dy, self.size)


def test_ResizeNearestNeighborGradAlignCornerT():
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float32)
    size = (4, 4)
    expect = np.array(
        [[[[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]]]]).astype(np.float32)
    rnn = ResizeNearestNeighborGradAlignCornerT(size=size)
    output = rnn(Tensor(dy))
    assert np.all(output.asnumpy() == expect)
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float16)
    size = (4, 4)
    expect = np.array(
        [[[[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]]]]).astype(np.float16)
    rnn = ResizeNearestNeighborGradAlignCornerT(size=size)
    output = rnn(Tensor(dy))
    assert np.all(output.asnumpy() == expect)
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.int32)
    size = (4, 4)
    expect = np.array(
        [[[[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]]]]).astype(np.int32)
    rnn = ResizeNearestNeighborGradAlignCornerT(size=size)
    output = rnn(Tensor(dy))
    assert np.all(output.asnumpy() == expect)


def test_ResizeNearestNeighborGradAlignCornerF():
    dy = np.array(
        [[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float32)
    size = (2, 2)
    expect = np.array([[[[4, 0], [0, 4]]]]).astype(np.float32)
    rnn = ResizeNearestNeighborGradAlignCornerF(size=size)
    output = rnn(Tensor(dy))
    assert np.all(output.asnumpy() == expect)
    dy = np.array(
        [[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float16)
    size = (2, 2)
    expect = np.array([[[[4, 0], [0, 4]]]]).astype(np.float16)
    rnn = ResizeNearestNeighborGradAlignCornerF(size=size)
    output = rnn(Tensor(dy))
    assert np.all(output.asnumpy() == expect)
    dy = np.array(
        [[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.int32)
    size = (2, 2)
    expect = np.array([[[[4, 0], [0, 4]]]]).astype(np.int32)
    rnn = ResizeNearestNeighborGradAlignCornerF(size=size)
    output = rnn(Tensor(dy))
    assert np.all(output.asnumpy() == expect)
