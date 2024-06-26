# Copyright 2022 Huawei Technologies Co., Ltd
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

import pytest
from tests.mark_utils import arg_mark
import numpy as np

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x, other):
        return x.fmod(other)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fmod(mode):
    """
    Feature: tensor.fmod
    Description: Verify the result of fmod
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    input_case_1 = ms.Tensor(np.array([-3., -2, -1, 1, 2, 3]), ms.float32)
    output_case_1 = net(input_case_1, 1.5)
    except_case_1 = np.array([-0.0000, -0.5000, -1.0000, 1.0000, 0.5000, 0.0000], dtype=np.float32)
    assert np.allclose(output_case_1.asnumpy(), except_case_1)

    input_case_2_1 = ms.Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
    input_case_2_2 = ms.Tensor(np.array([-1.5, -0.5, 1.5, 2.5, 3.5]), ms.float32)
    output_case_2 = net(input_case_2_1, input_case_2_2)
    except_case_2 = np.array([1.0000, 0.0000, 0.0000, 1.5000, 1.5000], dtype=np.float32)
    assert np.allclose(output_case_2.asnumpy(), except_case_2)
