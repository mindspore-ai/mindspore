# Copyright 2025 Huawei Technologies Co., Ltd
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

"""test feature: normalization of input int list for yaml-registered ops"""
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore as ms


class Net(ms.nn.Cell):
    def construct(self, x, size):
        return ms.ops.auto_generate.count_nonzero(x, size)

class TensorNet(ms.nn.Cell):
    def construct(self, x, size):
        return x.count_nonzero(size)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_list_normalization(mode):
    """
    Feature: int list normalization
    Description: list[int] and tuple[int] should support mixed int and Tensor
    Expectation: success
    """
    ms.set_context(mode=mode, jit_level="O0")
    x = ms.Tensor([[0, 1, 0], [1, 1, 0]], dtype=ms.float32)
    expected_output = 3
    net = Net()
    tensor_net = TensorNet()
    for dtype in [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8]:
        size = (0, ms.Tensor(1, dtype=dtype))
        output = net(x, size)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)
        output = tensor_net(x, size)
        assert np.allclose(output.asnumpy(), expected_output, rtol=0, atol=0)

    error_size = (0, ms.Tensor(1, dtype=ms.float32)) # float type not supported
    with pytest.raises((ValueError, TypeError)):
        output = net(x, error_size)
    with pytest.raises((ValueError, TypeError)):
        output = tensor_net(x, error_size)

    # non-single-element dim tensor not supported
    error_size = (0, ms.Tensor([1, 1], dtype=ms.int32))
    with pytest.raises((ValueError, TypeError)):
        output = net(x, error_size)
    with pytest.raises((ValueError, TypeError)):
        output = tensor_net(x, error_size)
