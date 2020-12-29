# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class SqueezeNet(Cell):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.squeeze = P.Squeeze()

    def construct(self, x):
        return self.squeeze(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_squeeze_shape_float32():
    x = np.ones(shape=[1, 2, 1, 1, 8, 3, 1]).astype(np.float32)
    expect = np.ones(shape=[2, 8, 3]).astype(np.float32)
    net = SqueezeNet()
    result = net(Tensor(x))
    assert np.allclose(result.asnumpy(), expect, rtol=1.e-4,
                       atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_squeeze_shape_int32():
    x = np.array([[7], [11]]).astype(np.int32)
    expect = np.array([7, 11]).astype(np.int32)
    net = SqueezeNet()
    result = net(Tensor(x))
    assert np.allclose(result.asnumpy(), expect, rtol=1.e-4,
                       atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_squeeze_shape_bool():
    x = np.array([[True], [False]]).astype(np.bool_)
    expect = np.array([True, False]).astype(np.bool_)
    net = SqueezeNet()
    result = net(Tensor(x))
    assert np.allclose(result.asnumpy(), expect, rtol=1.e-4,
                       atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_squeeze_shape_float64():
    x = np.random.random([1, 2, 1, 1, 8, 3, 1]).astype(np.float64)
    expect = np.squeeze(x)
    net = SqueezeNet()
    result = net(Tensor(x))
    print(result.asnumpy()[0][0], expect[0][0])
    assert np.allclose(result.asnumpy(), expect, rtol=1.e-4,
                       atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_squeeze_shape_uint16():
    x = np.random.random([1, 2, 1, 1, 8, 3, 1]).astype(np.uint16)
    expect = np.squeeze(x)
    net = SqueezeNet()
    result = net(Tensor(x))
    print(result.asnumpy()[0][0], expect[0][0])
    assert np.allclose(result.asnumpy(), expect, rtol=1.e-4,
                       atol=1.e-8, equal_nan=True)
