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
        for i in range(2):
            for _ in range(10):
                self.ta.write(index, value)
                index += 1
                value += 1
            if i == 0:
                self.ta.clear()
                index = 0
        v = self.ta.read(index-1)
        s = self.ta.stack()
        self.ta.close()
        return v, s


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensorarray():
    """
    Feature: TensorArray gpu TEST.
    Description: Test the function write, read, stack, clear, close in both graph and pynative mode.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    index = Tensor(0, mindspore.int64)
    value = Tensor(5, mindspore.int64)
    ta = TensorArrayNet(dtype=mindspore.int64, element_shape=())
    v, s = ta(index, value)
    expect_v = 24
    expect_s = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    assert np.allclose(s.asnumpy(), expect_s)
    assert v == expect_v

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    tb = nn.TensorArray(mindspore.int64, ())
    for i in range(5):
        tb.write(i, 99)
    v = tb.read(0)
    s = tb.stack()
    expect_v = 99
    expect_s = [99, 99, 99, 99, 99]
    assert np.allclose(s.asnumpy(), expect_s)
    assert np.allclose(v.asnumpy(), expect_v)
    tb_size = tb.size()
    assert np.allclose(tb_size.asnumpy(), 5)
    tb.clear()
    tb_size = tb.size()
    assert np.allclose(tb_size.asnumpy(), 0)
    tb.write(0, 88)
    v = tb.read(0)
    s = tb.stack()
    tb.close()
    expect_v = 88
    expect_s = [88]
    assert np.allclose(s.asnumpy(), expect_s)
    assert np.allclose(v.asnumpy(), expect_v)
    tc = nn.TensorArray(mindspore.float32, ())
    tc.write(5, 1.)
    s = tc.stack()
    expect_s = [0., 0., 0., 0., 0., 1.]
    assert np.allclose(s.asnumpy(), expect_s)
    tc.write(2, 1.)
    s = tc.stack()
    expect_s = [0., 0., 1., 0., 0., 1.]
    assert np.allclose(s.asnumpy(), expect_s)
    tc.close()
    td = nn.TensorArray(mindspore.bool_, ())
    td.write(1, Tensor(True, mindspore.bool_))
    s = td.stack()
    v = td.read(1)
    expect_s = [False, True]
    assert np.allclose(v.asnumpy(), expect_s[1])
    assert np.allclose(s.asnumpy(), expect_s)
    td.close()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_static_tensorarray():
    """
    Feature: TensorArray gpu TEST.
    Description: Test the static tensorarray.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    index = Tensor(0, mindspore.int64)
    value = Tensor(5, mindspore.int64)
    ta = TensorArrayNet(dtype=mindspore.int64, element_shape=(), is_dynamic_shape=False, size=12)
    v, s = ta(index, value)
    expect_v = 24
    expect_s = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 0]
    assert np.allclose(s.asnumpy(), expect_s)
    assert v == expect_v
