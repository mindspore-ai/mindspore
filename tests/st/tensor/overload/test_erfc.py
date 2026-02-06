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
from scipy import special
import pytest
import mindspore as ms
import mindspore.nn as nn
from tests.st.common.random_generator import generate_numpy_ndarray_by_randn
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x):
        output = x.erfc()
        return output


def erfc_expect_forward_func(x):
    return special.erfc(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
    'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_erfc(mode):
    """
    Feature: tensor.erfc
    Description: Verify the result of tensor.erfc
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x = generate_numpy_ndarray_by_randn((3, 4, 5, 6), np.float32, 'x')
    outputs = net(ms.Tensor(x))
    expect_output = erfc_expect_forward_func(x)
    assert np.allclose(outputs.asnumpy(), expect_output, rtol=1e-3, atol=1e-3)
