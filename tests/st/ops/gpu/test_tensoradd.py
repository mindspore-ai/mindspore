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
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

context.set_context(device_target='GPU')


class TensroAdd(nn.Cell):
    def __init__(self):
        super(TensroAdd, self).__init__()

        self.add = P.TensorAdd()

        self.x = Parameter(initializer(
            Tensor(np.random.randn(2, 0).astype(np.float32)), [2, 0]), name='x')
        self.y = Parameter(initializer(
            Tensor(np.random.randn(2, 1).astype(np.float32)), [2, 1]), name='y')

        self.x1 = Parameter(initializer(
            Tensor(np.arange(3).reshape(3).astype(np.float32)), [3]), name='x1')
        self.y1 = Parameter(initializer(
            Tensor(np.array([2]).astype(np.float32)), [1]), name='y1')

        self.x2 = Parameter(initializer(
            Tensor(np.arange(3 * 3 * 3 * 3).reshape(3, 3, 3, 3).astype(np.float32)), [3, 3, 3, 3]), name='x2')
        self.y2 = Parameter(initializer(
            Tensor(np.arange(3 * 3 * 3 * 3).reshape(3, 3, 3, 3).astype(np.float32)), [3, 3, 3, 3]), name='y2')

        self.x3 = Parameter(initializer(
            Tensor(np.arange(1 * 1 * 3 * 3).reshape(1, 1, 3, 3).astype(np.float32)), [1, 1, 3, 3]), name='x3')
        self.y3 = Parameter(initializer(
            Tensor(np.arange(3 * 3 * 3 * 3).reshape(3, 3, 3, 3).astype(np.float32)), [3, 3, 3, 3]), name='y3')

    @ms_function
    def construct(self):
        return (
            self.add(self.x, self.y), self.add(self.x1, self.y1), self.add(self.x2, self.y2),
            self.add(self.x3, self.y3))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_TensroAdd():
    add = TensroAdd()
    output = add()
    expect0 = np.array([])
    expect1 = np.array([2, 3, 4])
    expect2 = np.array(
        [[[[0., 2., 4.],
           [6., 8., 10.],
           [12., 14., 16.]],
          [[18., 20., 22.],
           [24., 26., 28.],
           [30., 32., 34.]],
          [[36., 38., 40.],
           [42., 44., 46.],
           [48., 50., 52.]]],
         [[[54., 56., 58.],
           [60., 62., 64.],
           [66., 68., 70.]],
          [[72., 74., 76.],
           [78., 80., 82.],
           [84., 86., 88.]],
          [[90., 92., 94.],
           [96., 98., 100.],
           [102., 104., 106.]]],
         [[[108., 110., 112.],
           [114., 116., 118.],
           [120., 122., 124.]],
          [[126., 128., 130.],
           [132., 134., 136.],
           [138., 140., 142.]],
          [[144., 146., 148.],
           [150., 152., 154.],
           [156., 158., 160.]]]])
    expect3 = np.array(
        [[[[0., 2., 4.],
           [6., 8., 10.],
           [12., 14., 16.]],
          [[9., 11., 13.],
           [15., 17., 19.],
           [21., 23., 25.]],
          [[18., 20., 22.],
           [24., 26., 28.],
           [30., 32., 34.]]],
         [[[27., 29., 31.],
           [33., 35., 37.],
           [39., 41., 43.]],
          [[36., 38., 40.],
           [42., 44., 46.],
           [48., 50., 52.]],
          [[45., 47., 49.],
           [51., 53., 55.],
           [57., 59., 61.]]],
         [[[54., 56., 58.],
           [60., 62., 64.],
           [66., 68., 70.]],
          [[63., 65., 67.],
           [69., 71., 73.],
           [75., 77., 79.]],
          [[72., 74., 76.],
           [78., 80., 82.],
           [84., 86., 88.]]]]
    )
    assert (output[0].asnumpy() == expect0).all()
    assert (output[1].asnumpy() == expect1).all()
    assert (output[2].asnumpy() == expect2).all()
    assert (output[3].asnumpy() == expect3).all()
