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
from mindspore.common.api import ms_function
from mindspore.ops import operations as P
from mindspore import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.AddN()

    @ms_function
    def construct(self, x, y, z):
        return self.add((x, y, z))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float32)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float32)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float32)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = [[[[0., 3., 6., 9.],
                       [12., 15., 18., 21.],
                       [24., 27., 30., 33.]],
                      [[36., 39., 42., 45.],
                       [48., 51., 54., 57.],
                       [60., 63., 66., 69.]],
                      [[72., 75., 78., 81.],
                       [84., 87., 90., 93.],
                       [96., 99., 102., 105.]]]]

    assert (output.asnumpy() == expect_result).all()

    # Test dynamic shape of addn.
    dyn_add = Net()
    input_x_dyn = Tensor(shape=[1, None, 3, 4], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[1, 3, None, 4], dtype=mstype.float32)
    input_z_dyn = Tensor(shape=[1, 3, 3, None], dtype=mstype.float32)
    dyn_add.set_inputs(input_x_dyn, input_y_dyn, input_z_dyn)
    output = add(Tensor(x), Tensor(y), Tensor(z))
    assert (output.asnumpy() == expect_result).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = np.array([[[[0., 3., 6., 9.],
                                [12., 15., 18., 21.],
                                [24., 27., 30., 33.]],
                               [[36., 39., 42., 45.],
                                [48., 51., 54., 57.],
                                [60., 63., 66., 69.]],
                               [[72., 75., 78., 81.],
                                [84., 87., 90., 93.],
                                [96., 99., 102., 105.]]]]).astype(np.float64)
    assert (output.asnumpy() == expect_result).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = np.array([[[[0., 3., 6., 9.],
                                [12., 15., 18., 21.],
                                [24., 27., 30., 33.]],
                               [[36., 39., 42., 45.],
                                [48., 51., 54., 57.],
                                [60., 63., 66., 69.]],
                               [[72., 75., 78., 81.],
                                [84., 87., 90., 93.],
                                [96., 99., 102., 105.]]]]).astype(np.float64)
    assert (output.asnumpy() == expect_result).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = np.array([[[[0., 3., 6., 9.],
                                [12., 15., 18., 21.],
                                [24., 27., 30., 33.]],
                               [[36., 39., 42., 45.],
                                [48., 51., 54., 57.],
                                [60., 63., 66., 69.]],
                               [[72., 75., 78., 81.],
                                [84., 87., 90., 93.],
                                [96., 99., 102., 105.]]]]).astype(np.int64)
    assert (output.asnumpy() == expect_result).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = np.array([[[[0., 3., 6., 9.],
                                [12., 15., 18., 21.],
                                [24., 27., 30., 33.]],
                               [[36., 39., 42., 45.],
                                [48., 51., 54., 57.],
                                [60., 63., 66., 69.]],
                               [[72., 75., 78., 81.],
                                [84., 87., 90., 93.],
                                [96., 99., 102., 105.]]]]).astype(np.int64)
    assert (output.asnumpy() == expect_result).all()
