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
        return self.x.matmul(*args, **kwargs)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_matmul(mode):
    """
    Feature: tensor.matmul
    Description: Verify the result of tensor.matmul
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    # test1: no broadcast
    x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4), ms.float32)
    other = Tensor(np.arange(4 * 5).reshape(4, 5), ms.float32)
    net = Net(x)
    output_x = net(other)
    expect_x = Tensor(np.array([[[70, 76, 82, 88, 94],
                                 [190, 212, 234, 256, 278],
                                 [310, 348, 386, 424, 462]],
                                [[430, 484, 538, 592, 646],
                                 [550, 620, 690, 760, 830],
                                 [670, 756, 842, 928, 1014]]]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test2: broadcast
    x = Tensor(np.ones([1, 2]), ms.float32)
    other = Tensor(np.ones([2,]), ms.float32)
    net = Net(x)
    output_x = net(other)
    expect_x = Tensor(np.array([2]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())
