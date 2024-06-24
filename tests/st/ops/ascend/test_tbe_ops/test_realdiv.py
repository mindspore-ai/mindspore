# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.realdiv = P.RealDiv()

    @jit
    def construct(self, x1, x2):
        return self.realdiv(x1, x2)


arr_x1 = np.random.randn(3, 4).astype(np.float32)
arr_x2 = np.random.randn(3, 4).astype(np.float32)


def test_net():
    realdiv = Net()
    output = realdiv(Tensor(arr_x1), Tensor(arr_x2))
    print(arr_x1)
    print(arr_x2)
    print(output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_realdiv_bf16(mode):
    """
    Feature: Test realdiv forward tensor api.
    Description: Test realdiv for bfloat16.
    Expectation: the result match with the expected result.
    :return:
    """
    context.set_context(mode=mode, device_target="Ascend")
    realdiv = Net()
    inputa_bf16 = Tensor([0.2, 0.74, 0.04], mstype.bfloat16)
    inputb_bf16 = Tensor([0.102, 0.55, 0.88], mstype.bfloat16)
    output = realdiv(inputa_bf16, inputb_bf16).float().asnumpy()
    expected = np.array([1.9617225, 1.3404255, 0.04555555]).astype(np.float32)
    assert np.allclose(output, expected, rtol=0.007, atol=0.007)
