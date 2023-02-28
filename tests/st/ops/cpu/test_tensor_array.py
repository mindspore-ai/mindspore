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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor

class TensorArrayNet(nn.Cell):
    def __init__(self, dtype, element_shape, is_dynamic_shape=True, size=0):
        super(TensorArrayNet, self).__init__()
        self.ta = nn.TensorArray(dtype, element_shape, is_dynamic_shape, size)

    def construct(self, index, value):
        self.ta.write(index, value)
        v = self.ta.read(index)
        s = self.ta.stack()
        self.ta.close()
        return v, s


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensorarray():
    """
    Feature: TensorArray gpu TEST.
    Description: Test the function write, read, stack, clear, close in both graph and pynative mode.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    index = Tensor(0, mindspore.int64)
    value = Tensor(5, mindspore.int64)
    ta = TensorArrayNet(dtype=mindspore.int64, element_shape=())
    v, s = ta(index, value)
    expect_v = 5
    expect_s = [5]
    assert np.allclose(s.asnumpy(), expect_s)
    assert v == expect_v

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    ta = nn.TensorArray(mindspore.int64, ())
    for i in range(5):
        ta.write(i, 99)
    v = ta.read(0)
    s = ta.stack()
    expect_v = 99
    expect_s = [99, 99, 99, 99, 99]
    assert np.allclose(s.asnumpy(), expect_s)
    assert np.allclose(v.asnumpy(), expect_v)
    ta_size = ta.size()
    assert np.allclose(ta_size.asnumpy(), 5)
    ta.clear()
    ta_size = ta.size()
    assert np.allclose(ta_size.asnumpy(), 0)
    ta.write(0, 88)
    v = ta.read(0)
    s = ta.stack()
    ta.close()
    expect_v = 88
    expect_s = [88]
    assert np.allclose(s.asnumpy(), expect_s)
    assert np.allclose(v.asnumpy(), expect_v)
    ta = nn.TensorArray(mindspore.float32, ())
    ta.write(5, 1.)
    s = ta.stack()
    expect_s = [0., 0., 0., 0., 0., 1.]
    assert np.allclose(s.asnumpy(), expect_s)
    ta.write(2, 1.)
    s = ta.stack()
    expect_s = [0., 0., 1., 0., 0., 1.]
    assert np.allclose(s.asnumpy(), expect_s)
    ta.close()
    ta = nn.TensorArray(mindspore.bool_, ())
    ta.write(1, Tensor(True, mindspore.bool_))
    s = ta.stack()
    v = ta.read(1)
    expect_s = [False, True]
    assert np.allclose(v.asnumpy(), expect_s[1])
    assert np.allclose(s.asnumpy(), expect_s)
    ta.close()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_static_tensorarray():
    """
    Feature: TensorArray cpu TEST.
    Description: Test the static tensorarray.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    index = Tensor(0, mindspore.int64)
    value = Tensor(5, mindspore.int64)
    ta = TensorArrayNet(dtype=mindspore.int64, element_shape=(), is_dynamic_shape=False, size=12)
    v, s = ta(index, value)
    expect_v = 5
    expect_s = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(s.asnumpy(), expect_s)
    assert v == expect_v
