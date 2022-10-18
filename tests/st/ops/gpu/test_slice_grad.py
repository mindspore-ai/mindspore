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
from mindspore.common.api import jit
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class SliceGrad(nn.Cell):
    def __init__(self):
        super(SliceGrad, self).__init__()
        self.slice_grad = G.SliceGrad()

    @jit
    def construct(self, dy, x):
        return self.slice_grad(dy, x, (0, 1, 0), (2, 1, 3))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice():
    x = Tensor(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]]).astype(np.float32))
    dy = Tensor(np.array([[[3., 1., 2.]], [[4., 1., 4.]]]).astype(np.float32))
    slice_grad = SliceGrad()
    output = slice_grad(dy, x)
    expect = [[[0., 0., 0.],
               [3., 1., 2.]],
              [[0., 0., 0.],
               [4., 1., 4.]],
              [[0., 0., 0.],
               [0., 0., 0.]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice_float64():
    x = Tensor(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]]).astype(np.float64))
    dy = Tensor(np.array([[[3., 1., 2.]], [[4., 1., 4.]]]).astype(np.float64))
    slice_grad = SliceGrad()
    output = slice_grad(dy, x)
    expect = np.array([[[0., 0., 0.],
                        [3., 1., 2.]],
                       [[0., 0., 0.],
                        [4., 1., 4.]],
                       [[0., 0., 0.],
                        [0., 0., 0.]]]).astype(np.float64)
    assert (output.asnumpy() == expect).all()


class SliceGrad7D(nn.Cell):
    def __init__(self):
        super(SliceGrad7D, self).__init__()
        self.slice_grad = G.SliceGrad()

    @jit
    def construct(self, dy, x):
        return self.slice_grad(dy, x, (1, 0, 2, 0, 0, 0, 0), (1, 2, 1, 1, 1, 1, 2))


def test_slice_grad_7d():
    """
    Feature: SliceGrad
    Description: test SliceGrad with 7D input
    Expectation: the output is as expected
    """
    x = Tensor(np.array([[[[[[[3, 4]]]], [[[[8, 9]]]], [[[[3, 2]]]]],
                          [[[[[4, 4]]]], [[[[8, 6]]]], [[[[1, 7]]]]]],
                         [[[[[[7, 2]]]], [[[[3, 7]]]], [[[[5, 8]]]]],
                          [[[[[3, 2]]]], [[[[6, 0]]]], [[[[7, 6]]]]]]]).astype(np.int32))
    dy = Tensor(np.arange(1 * 2 * 1 * 1 * 1 * 1 * 2).reshape(1, 2, 1, 1, 1, 1, 2).astype(np.int32))
    slice_grad = SliceGrad7D()
    output = slice_grad(dy, x)
    expect = np.zeros((2, 2, 3, 1, 1, 1, 2))
    expect[1:2, 0:2, 2:3, 0:1, 0:1, 0:1, 0:2] = dy
    print("output:\n", output)
    assert (output.asnumpy() == expect).all()


if __name__ == '__main__':
    test_slice()
    test_slice_float64()
    test_slice_grad_7d()
