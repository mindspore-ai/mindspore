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
""" test grad in MSAdapter. """

import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.ops import grad
from tests.st.compiler.ms_adapter import Tensor
from tests.mark_utils import arg_mark


ms.set_context(mode=ms.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_adapter_grad():
    """
    Feature: MSAdapter
    Description: Test grad scenario of MSAdapter
    Expectation: No exception
    """
    class Net(nn.Cell):
        def construct(self, x, y, z):
            return x * y * z

    x = Tensor([1, 2], dtype=mstype.int32)
    y = Tensor([-2, 3], dtype=mstype.int32)
    z = Tensor([0, 3], dtype=mstype.int32)
    net = Net()
    output = grad(net, grad_position=(1, 2))(x, y, z)

    grad_y = Tensor([0, 6], dtype=mstype.int32)
    grad_z = Tensor([-2, 6], dtype=mstype.int32)
    assert np.all(output[0].asnumpy() == grad_y.asnumpy())
    assert np.all(output[1].asnumpy() == grad_z.asnumpy())
