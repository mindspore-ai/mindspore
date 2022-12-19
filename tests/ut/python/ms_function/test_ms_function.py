# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore import context, Tensor
from mindspore.common.api import jit

grad_all = C.GradOperation(get_all=True)


class CellBprop(nn.Cell):
    def construct(self, x, y):
        return 2 * x * x + y * y

    @jit
    def bprop(self, x, y, out, dout):
        return dout, 2 * y


def test_cell_bprop_grad():
    input_x = Tensor(np.random.randn(2, 2).astype(np.float32))
    input_y = Tensor(np.random.randn(2, 2).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net = CellBprop()
    with pytest.raises(RuntimeError):
        grad_all(net)(input_x, input_y)
