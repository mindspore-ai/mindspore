# Copyright 2023 Huawei Technologies Co., Ltd
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
    def construct(self, x):
        return x.floor()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_floor(mode):
    """
    Feature: tensor.floor
    Description: Verify the result of floor
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    input_case = ms.Tensor(np.array([1.1, 2.5, -1.5]), ms.float32)
    output_case = net(input_case)
    except_case = np.array([1., 2., -2.], dtype=np.float32)
    assert output_case.asnumpy().dtype == np.float32
    assert np.allclose(output_case.asnumpy(), except_case)
