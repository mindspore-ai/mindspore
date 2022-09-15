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

import numpy as np
import pytest

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.IsFinite()

    def construct(self, x):
        return self.ops(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    x0 = Tensor(np.array([np.log(-1), 0.4, np.log(0)]).astype(np.float16))
    x1 = Tensor(np.array([np.log(-1), 0.4, np.log(0)]).astype(np.float32))
    x2 = Tensor(np.array([np.log(-1), 0.4, np.log(0)]).astype(np.float64))
    x3 = Tensor(np.array([4, 1, -5]).astype(np.int8))
    x4 = Tensor(np.array([4, 1, -5]).astype(np.int16))
    x5 = Tensor(np.array([4, 1, -5]).astype(np.int32))
    x6 = Tensor(np.array([4, 1, -5]).astype(np.int64))
    x7 = Tensor(np.array([4, 1, -5]).astype(np.uint8))
    x8 = Tensor(np.array([4, 1, -5]).astype(np.uint16))
    x9 = Tensor(np.array([4, 1, -5]).astype(np.uint32))
    x10 = Tensor(np.array([4, 1, -5]).astype(np.uint64))
    x11 = Tensor(np.array([False, True, False]).astype(np.bool_))

    net = Net()
    out = net(x0).asnumpy()
    expect = [False, True, False]
    assert np.all(out == expect)

    out = net(x1).asnumpy()
    expect = [False, True, False]
    assert np.all(out == expect)

    out = net(x2).asnumpy()
    expect = [False, True, False]
    assert np.all(out == expect)

    out = net(x3).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)

    out = net(x4).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)

    out = net(x5).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)

    out = net(x6).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)

    out = net(x7).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)

    out = net(x8).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)

    out = net(x9).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)

    out = net(x10).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)

    out = net(x11).asnumpy()
    expect = [True, True, True]
    assert np.all(out == expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_finite_cpu_dynamic_shape():
    """
    Feature: test FloatStatus op on CPU.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, ms.float32))
    except_shape = (1, 32, 9, 9)
    assert output.asnumpy().shape == except_shape
