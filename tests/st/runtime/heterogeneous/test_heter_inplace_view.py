# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test network when heterogeneous_excutor."""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops.auto_generate import TransposeView
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark
import pytest

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_inplace_ops_with_view_input_in_heterogeneous():
    """
    Feature: Inplace operator doesn't accept a view input
    Description: Runtime throws an exception when an inplace operator uses a view input
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        """
        TestNet for inplace ops with view input in heter scene.
        """
        def __init__(self):
            super().__init__()
            self.assign_add = P.AssignAdd()
            self.assign_add.set_device("CPU")

        def construct(self, tensor, value):
            view_tensor = TransposeView()(tensor, (1, 0))
            self.assign_add(view_tensor, value)
            return view_tensor

    def func():
        tensor = Tensor(np.ones((3, 2), dtype=np.float32))
        value_tensor = Tensor(np.ones((2, 3), dtype=np.float32))
        net = TestNet()
        net(tensor, value_tensor)

    with pytest.raises(RuntimeError) as err:
        func_jit = ms.jit(func, backend="ms_backend")
        func_jit()
    assert "Not support non-contiguous input" in str(err.value)
