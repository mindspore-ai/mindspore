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
Tests for Tensor.masked_fill.
"""
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, mask, value):
        return x.masked_fill(mask=mask, value=value)


_MS_TYPE = [ms.bool_, ms.int8, ms.int32, ms.int64, ms.float16, ms.float32]
_NP_TYPE = [np.bool_, np.int8, np.int32, np.int64, np.float16, np.float32]


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_masked_fill(mode):
    """
    Feature: tensor.masked_fill
    Description: Verify the result of masked_fill
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    for np_type, ms_type in zip(_NP_TYPE, _MS_TYPE):
        input_x = Tensor(np.array([1., 2., 3., 4.]), ms_type)
        mask = Tensor(np.array([True, True, False, True]), ms.bool_)
        value = Tensor(0.5, dtype=ms_type)
        output = net(input_x, mask, value)
        expected = np.array([0.5, 0.5, 3., 0.5], dtype=np_type)
        assert np.allclose(output.asnumpy(), expected)
