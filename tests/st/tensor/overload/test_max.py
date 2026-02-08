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
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):

    def __init__(self, x):
        super(Net, self).__init__()
        self.x = x

    def construct(self, *args, **kwargs):
        return self.x.max(*args, **kwargs)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_max(mode):
    """
    Feature: tensor.max
    Description: Verify the result of tensor.max
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.arange(4).reshape((2, 2)).astype('float32'))
    net = Net(x)

    # test 1: using default args
    output_x = net()
    expect_x = 3.0
    assert output_x == expect_x

    # test 2: using positional args
    output_x = net(0, True, initial=None)
    expect_x = Tensor(np.array([[2, 3]]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 3: using kv args
    output_x, indices = net(axis=0, return_indices=True)
    expect_x = Tensor(np.array([2, 3]), ms.float32)
    expect_indices = Tensor(np.array([1, 1]), ms.int32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())
    assert np.allclose(indices.asnumpy(), expect_indices.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_max_dim(mode):
    """
    Feature: Functional.
    Description: Test tensor method overload x.max(dim, keepdim=False)
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = ms.Tensor(np.arange(4).reshape((2, 2)).astype(np.float32))
    net = Net(x)
    dim = 0
    keepdim = False
    if mode == 0:
        output = net(dim, keepdim)
        assert np.allclose(output.asnumpy(), np.array([2.0, 3.0]))
    else:
        output, index = net(dim, keepdim)
        assert np.allclose(output.asnumpy(), np.array([2.0, 3.0]))
        assert np.allclose(index.asnumpy(), np.array([1, 1]))
