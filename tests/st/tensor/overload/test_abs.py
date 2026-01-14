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
from mindspore import Tensor, nn


class Net(nn.Cell):
    def construct(self, x):
        return abs(x)


class Net2(nn.Cell):
    def construct(self, x):
        return x.abs()


_MS_TYPES = [ms.float16, ms.float32, ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]
_NP_TYPES = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.03)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_abs(mode):
    """
    Feature: tensor.__abs__
    Description: Verify the result of absolute
    Expectation: success
    """
    ms.set_context(mode=mode)

    for ms_type, np_type in zip(_MS_TYPES, _NP_TYPES):
        x = Tensor(np.array([-1.0, 1.0, 0.0, -1.1]), ms_type)
        net = Net()
        net2 = Net2()
        output = net(x)
        output2 = net2(x)
        expect_output = np.array([1., 1., 0., 1.1], dtype=np_type)
        assert np.allclose(output.asnumpy(), expect_output)
        assert np.allclose(output2.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.03)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_absolute(mode):
    """
    Feature: tensor.__abs__
    Description: Verify the result of absolute
    Expectation: success
    """
    ms.set_context(mode=mode)
    for ms_type, np_type in zip(_MS_TYPES, _NP_TYPES):
        x = Tensor(np.array(-1), ms_type)
        net = Net()
        net2 = Net2()
        output = net(x)
        output2 = net2(x)
        expect_output = np.array(1, dtype=np_type)
        assert np.allclose(output.asnumpy(), expect_output)
        assert np.allclose(output2.asnumpy(), expect_output)
