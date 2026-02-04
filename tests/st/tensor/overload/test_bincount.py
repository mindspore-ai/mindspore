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

import mindspore as ms
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils

@test_utils.run_with_cell
def bincount_forward_func(x, weights=None, minlength=0):
    return x.bincount(weights, minlength)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_tensor_bincount(mode):
    """
    Feature: Tensor.bincount
    Description: Verify the result of Tensor.bincount
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = Tensor([2, 4, 1, 0, 0], ms.int64)
    weights = Tensor([0, 0.25, 0.5, 0.75, 1], ms.float32)
    expect_output = Tensor([1.75, 0.5, 0, 0, 0.25], ms.float32)
    output = bincount_forward_func(x, weights, minlength=5)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
