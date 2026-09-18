# Copyright 2024 Huawei Technologies Co., Ltd
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

"""
Tests for  Tensor.allclose
"""

import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, axis=(), keep_dims=False):
        return x.any(axis=axis, keep_dims=keep_dims)


class Net2(nn.Cell):
    def construct(self, x):
        return x.any()


class Net3(nn.Cell):
    def construct(self, x, dim, keepdim=False):
        return x.any(dim=dim, keepdim=keepdim)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', ],
          level_mark='level0',
          card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_any(mode):
    """
    Feature: tensor.any
    Description: Verify the result of any
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()

    x = Tensor(np.array([[True, False], [False, False]]))
    output = net(x, keep_dims=True)
    assert np.allclose(output.asnumpy(), np.array([[True]]))
    assert np.allclose(output.shape, (1, 1))

    output2 = net(x, axis=0)
    assert np.allclose(output2.asnumpy(), np.array([True, False]))
    assert np.allclose(output2.shape, (2,))

    output3 = net(x, axis=1)
    assert np.allclose(output3.asnumpy(), np.array([True, False]))
    assert np.allclose(output3.shape, (2,))

    output_2 = Net2()(x)
    assert np.allclose(output_2.asnumpy(), np.array([True]))
    assert np.allclose(output_2.shape, ())

    net3 = Net3()
    output2_2 = net3(x, dim=0)
    assert np.allclose(output2_2.asnumpy(), np.array([True, False]))
    assert np.allclose(output2_2.shape, (2,))

    output3_2 = net3(x, dim=1)
    assert np.allclose(output3_2.asnumpy(), np.array([True, False]))
    assert np.allclose(output3_2.shape, (2,))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', ],
          level_mark='level0',
          card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize('dtype', [np.int16, np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_any_multi_byte_dtype_axis_reduce(mode, dtype):
    """
    Feature: tensor.any
    Description: Verify any of a multi-byte dtype input tests each element instead of its raw bytes.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    # Column 0 and column 2 hold a non-zero element, column 1 is all zeros.
    x = Tensor(np.array([[1, 0, 3], [0, 0, 0]], dtype))
    output = net(x, axis=0)
    assert np.array_equal(output.asnumpy(), np.array([True, False, True]))
    assert output.shape == (3,)
