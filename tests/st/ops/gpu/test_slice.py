# Copyright 2019-2021 Huawei Technologies Co., Ltd
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


class Slice(nn.Cell):
    def __init__(self):
        super(Slice, self).__init__()
        self.slice = P.Slice()

    def construct(self, x):
        return self.slice(x, (0, 1, 0), (2, 1, 3))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice():
    x = Tensor(
        np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]).astype(np.float32))
    expect = [[[2., -2., 2.]],
              [[4., -4., 4.]]]

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    slice_op = Slice()
    output = slice_op(x)
    assert (output.asnumpy() == expect).all()


class SliceNet(nn.Cell):
    def __init__(self):
        super(SliceNet, self).__init__()
        self.slice = P.Slice()

    def construct(self, x):
        return self.slice(x, (0, 11, 0, 0), (32, 7, 224, 224))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice_4d():
    x_np = np.random.randn(32, 24, 224, 224).astype(np.float32)
    output_np = x_np[:, 11:18, :, :]

    x_ms = Tensor(x_np)
    net = SliceNet()
    output_ms = net(x_ms)

    assert (output_ms.asnumpy() == output_np).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice_float64():
    x = Tensor(
        np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]).astype(np.float64))
    expect = np.array([[[2., -2., 2.]],
                       [[4., -4., 4.]]]).astype(np.float64)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    slice_op = Slice()
    output = slice_op(x)
    assert (output.asnumpy() == expect).all()
