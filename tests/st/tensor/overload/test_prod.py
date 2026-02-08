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
    def construct(self, x, dim=None, keepdim=False):
        return x.prod(dim, keepdim)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_prod(mode):
    """
    Feature: Tensor.negative
    Description: Verify the result of Tensor.negative
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                         [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                         [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), ms.float32)
    net = Net()
    output_x = net(x)
    expect_x = Tensor(np.array([2.2833798e+33]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    output_x = net(x, 0, True)
    expect_x = Tensor(np.array([[[28, 28, 28, 28, 28, 28],
                                 [80, 80, 80, 80, 80, 80],
                                 [162, 162, 162, 162, 162, 162]]]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    output_x = net(x, 1, True)
    expect_x = Tensor(np.array([[[6, 6, 6, 6, 6, 6]],
                                [[120, 120, 120, 120, 120, 120]],
                                [[504, 504, 504, 504, 504, 504]]]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    output_x = net(x, 2, True)
    expect_x = Tensor(np.array([[[1.00000e+00],
                                 [6.40000e+01],
                                 [7.29000e+02]],
                                [[4.09600e+03],
                                 [1.56250e+04],
                                 [4.66560e+04]],
                                [[1.17649e+05],
                                 [2.62144e+05],
                                 [5.31441e+05]]]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())
