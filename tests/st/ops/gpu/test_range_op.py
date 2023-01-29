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

import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops, jit
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class RangeNet(nn.Cell):
    def __init__(self, maxlen=50):
        super(RangeNet, self).__init__()
        self.range = P.Range(maxlen)

    def construct(self, start, limit, delta):
        return self.range(start, limit, delta)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_precision_end_equals_last_element():
    range_net = RangeNet(100)
    ms_out = range_net(Tensor(1000.04, mstype.float32),
                       Tensor(1001.04, mstype.float32),
                       Tensor(0.01, mstype.float32)).asnumpy()
    np_expected = np.arange(1000.04, 1001.04, 0.01, dtype=np.float32)
    np.testing.assert_allclose(ms_out, np_expected, rtol=1e-5)

    range_net = RangeNet(1000)
    ms_out = range_net(Tensor(100, mstype.float32),
                       Tensor(101, mstype.float32),
                       Tensor(0.001, mstype.float32)).asnumpy()
    np_expected = np.arange(100, 101, 0.001, dtype=np.float32)
    np.testing.assert_allclose(ms_out, np_expected, rtol=1e-5)

    range_net = RangeNet(799900)
    ms_out = range_net(Tensor(1, mstype.float32),
                       Tensor(8000, mstype.float32),
                       Tensor(0.01, mstype.float32)).asnumpy()
    np_expected = np.arange(1, 8000, 0.01, dtype=np.float32)
    np.testing.assert_allclose(ms_out, np_expected, rtol=1e-5)

    range_net = RangeNet(53)
    ms_out = range_net(Tensor(-12000, mstype.float32),
                       Tensor(-12053, mstype.float32),
                       Tensor(-1, mstype.float32)).asnumpy()
    np_expected = np.arange(-12000, -12053, -1, dtype=np.float32)
    np.testing.assert_allclose(ms_out, np_expected, rtol=1e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_int():
    range_net = RangeNet()
    ms_out = range_net(Tensor(2, mstype.int32), Tensor(5, mstype.int32), Tensor(1, mstype.int32)).asnumpy()
    np_expected = np.array([2, 3, 4])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(-24, mstype.int32), Tensor(1, mstype.int32), Tensor(4, mstype.int32)).asnumpy()
    np_expected = np.array([-24, -20, -16, -12, -8, -4, 0])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(8, mstype.int32), Tensor(1, mstype.int32), Tensor(-1, mstype.int32)).asnumpy()
    np_expected = np.array([8, 7, 6, 5, 4, 3, 2])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(3, mstype.int32), Tensor(-11, mstype.int32), Tensor(-5, mstype.int32)).asnumpy()
    np_expected = np.array([3, -2, -7])
    np.testing.assert_array_equal(ms_out, np_expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_float():
    range_net = RangeNet()
    ms_out = range_net(Tensor(2.3, mstype.float32), Tensor(5.5, mstype.float32), Tensor(1.2, mstype.float32)).asnumpy()
    np_expected = np.array([2.3, 3.5, 4.7])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(-4, mstype.float32), Tensor(-1, mstype.float32), Tensor(1.5, mstype.float32)).asnumpy()
    np_expected = np.array([-4.0, -2.5])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(8.0, mstype.float32), Tensor(1.0, mstype.float32), Tensor(-1.0, mstype.float32)).asnumpy()
    np_expected = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(1.5, mstype.float32), Tensor(-1, mstype.float32), Tensor(-18.9, mstype.float32)).asnumpy()
    np_expected = np.array([1.5])
    np.testing.assert_array_almost_equal(ms_out, np_expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_int64():
    """
    Feature: test Range op on GPU.
    Description: test the Range when input is int64.
    Expectation: result is right.
    """
    range_net = RangeNet()
    ms_out = range_net(Tensor(2, mstype.int64), Tensor(5, mstype.int64), Tensor(1, mstype.int64)).asnumpy()
    np_expected = np.array([2, 3, 4])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(-24, mstype.int64), Tensor(1, mstype.int64), Tensor(4, mstype.int64)).asnumpy()
    np_expected = np.array([-24, -20, -16, -12, -8, -4, 0])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(8, mstype.int64), Tensor(1, mstype.int64), Tensor(-1, mstype.int64)).asnumpy()
    np_expected = np.array([8, 7, 6, 5, 4, 3, 2])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(3, mstype.int64), Tensor(-11, mstype.int64), Tensor(-5, mstype.int64)).asnumpy()
    np_expected = np.array([3, -2, -7])
    np.testing.assert_array_equal(ms_out, np_expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_float64():
    """
    Feature: test Range op on GPU.
    Description: test the Range when input is float64.
    Expectation: result is right.
    """
    range_net = RangeNet()
    ms_out = range_net(Tensor(2.3, mstype.float64), Tensor(5.5, mstype.float64), Tensor(1.2, mstype.float64)).asnumpy()
    np_expected = np.array([2.3, 3.5, 4.7])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(-4, mstype.float64), Tensor(-1, mstype.float64), Tensor(1.5, mstype.float64)).asnumpy()
    np_expected = np.array([-4.0, -2.5])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(8.0, mstype.float64), Tensor(1.0, mstype.float64), Tensor(-1.0, mstype.float64)).asnumpy()
    np_expected = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(1.5, mstype.float64), Tensor(-1, mstype.float64), Tensor(-18.9, mstype.float64)).asnumpy()
    np_expected = np.array([1.5])
    np.testing.assert_array_almost_equal(ms_out, np_expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_invalid_max_output_length():
    with pytest.raises(ValueError):
        _ = P.Range(0)
        _ = P.Range(-1)
        _ = P.Range(None)
        _ = P.Range('5')


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_invalid_input():
    with pytest.raises(ValueError) as info:
        range_net = RangeNet()
        _ = range_net(Tensor(0, mstype.int32), Tensor(5, mstype.int32), Tensor(0, mstype.int32)).asnumpy()
    assert "delta cannot be equal to zero" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        range_net = RangeNet(2)
        _ = range_net(Tensor(2, mstype.int32), Tensor(5, mstype.int32), Tensor(1, mstype.int32)).asnumpy()
    assert "number of elements in the output exceeds maxlen" in str(info.value)

    with pytest.raises(ValueError) as info:
        range_net = RangeNet()
        _ = range_net(Tensor(20, mstype.int32), Tensor(5, mstype.int32), Tensor(1, mstype.int32)).asnumpy()
    assert "delta cannot be positive when limit < start" in str(info.value)

    with pytest.raises(ValueError) as info:
        range_net = RangeNet()
        _ = range_net(Tensor(2, mstype.int32), Tensor(5, mstype.int32), Tensor(-4, mstype.int32)).asnumpy()
    assert "delta cannot be negative when limit > start" in str(info.value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_functional():
    """
    Feature: functional range.
    Description: Test functional interface range.
    Expectation: success
    """
    start = Tensor(0, mstype.int32)
    limit = Tensor(10, mstype.int32)
    delta = Tensor(4, mstype.int32)
    output = ops.range(start, limit, delta)
    assert np.all(output.asnumpy() == np.array([0, 4, 8]))


@jit
def range_fn(x, y, z, a):
    return ops.range(x, y, z) + a


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_vmap():
    """
    Feature: Ops range with vmap.
    Description: Test ops range in vmap.
    Expectation: success
    """
    start = Tensor(0, mstype.int32)
    limit = Tensor(10, mstype.int32)
    delta = Tensor(4, mstype.int32)
    a = Tensor([[1, 1, 1], [1, 1, 1]])
    vmap_range = ops.vmap(range_fn, (None, None, None, 0), 0)
    output = vmap_range(start, limit, delta, a)
    assert np.all(output.asnumpy() == np.array([[1, 5, 9], [1, 5, 9]]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_vmap_wrong_in_axis():
    """
    Feature: Ops range with vmap.
    Description: Test ops range in vmap with wrong in axis value.
    Expectation: ValueError.
    """
    start = Tensor([0, 1], mstype.int32)
    limit = Tensor(10, mstype.int32)
    delta = Tensor(4, mstype.int32)
    a = Tensor([[1, 1, 1], [1, 1, 1]])
    vmap_range = ops.vmap(range_fn, (0, None, None, 0), 0)
    with pytest.raises(ValueError) as ex:
        vmap_range(start, limit, delta, a)
    assert "For operator Range, all axis for inputs should be None" in str(ex.value)
