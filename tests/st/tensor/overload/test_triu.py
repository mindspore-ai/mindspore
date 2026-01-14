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
        return self.x.triu(*args, **kwargs)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_triu(mode):
    """
    Feature: tensor.triu
    Description: Verify the result of tensor.triu
    Expectation: success
    """
    # test 1: diagonal is default
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]]))
    net = Net(x)
    output_x = net()
    expect_x = Tensor(np.array([[1, 2, 3, 4],
                                [0, 6, 7, 8],
                                [0, 0, 12, 13],
                                [0, 0, 0, 17]]))
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 2: diagonal is positional args
    x = Tensor(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]]))
    net = Net(x)
    output_x = net(1)
    expect_x = Tensor(np.array([[0, 2, 3, 4],
                                [0, 0, 7, 8],
                                [0, 0, 0, 13],
                                [0, 0, 0, 0]]))
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 3: diagonal is kv args
    x = Tensor(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]]))
    net = Net(x)
    output_x = net(diagonal=-1)
    expect_x = Tensor(np.array([[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [0, 11, 12, 13],
                                [0, 0, 16, 17]]))
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())
