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

import random
from functools import reduce
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetArgmax(nn.Cell):
    def __init__(self, axis=0):
        super(NetArgmax, self).__init__()
        self.argmax = ops.Argmax(axis=axis, output_type=mstype.int32)

    def construct(self, x):
        return self.argmax(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmax_1d():
    x = Tensor(np.array([1., 20., 5.]).astype(np.float32))
    Argmax = NetArgmax(axis=0)
    output = Argmax(x)
    expect = np.array([1]).astype(np.float32)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmax_2d():
    x = Tensor(np.array([[1., 20., 5.],
                         [67., 8., 9.],
                         [130., 24., 15.]]).astype(np.float32))
    Argmax_axis_0 = NetArgmax(axis=0)
    output = Argmax_axis_0(x)
    expect = np.array([2, 2, 2]).astype(np.float32)
    assert (output.asnumpy() == expect).all()
    Argmax_axis_1 = NetArgmax(axis=1)
    output = Argmax_axis_1(x)
    expect = np.array([1, 0, 0]).astype(np.float32)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmax_high_dims():
    for dim in range(3, 10):
        shape = np.random.randint(1, 10, size=dim)
        x = np.random.randn(reduce(lambda x, y: x * y, shape)).astype(np.float32)
        x = x.reshape(shape)

        rnd_axis = random.randint(-dim + 1, dim - 1)
        Argmax = NetArgmax(axis=rnd_axis)
        ms_output = Argmax(Tensor(x))
        np_output = np.argmax(x, axis=rnd_axis)
        assert (ms_output.asnumpy() == np_output).all()
