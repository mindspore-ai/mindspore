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
# ============================================================================

"""Test for after while after if."""

import mindspore as ms
from mindspore import Tensor, ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_for_after_while_after_if():
    """
    Feature: Control flow.
    Description: Nested control flow (for-after-while-after-if) in jit.
    Expectation: Executes correctly and gradients are computable.
    """
    i = Tensor(16, ms.int32)

    def net(x, y):
        if x > y:
            x = x - y
        while x != i:
            x = x * 2
            continue
        for _ in range(1, 3):
            y = y * 2
            return x - y
        return x

    input_x = Tensor(4, ms.int32)
    input_y = Tensor(2, ms.int32)

    # input_x > input_y
    res = ms.jit(net)(input_x, input_y)
    assert res == Tensor(12, ms.int32)
    grad_res = ms.jit(ops.GradOperation(get_all=True)(net))(input_x, input_y)
    assert len(grad_res) == 2

    # input_x < input_y
    input_y = Tensor(6, ms.int32)
    res = ms.jit(net)(input_x, input_y)
    assert res == Tensor(4, ms.int32)
    grad_res = ms.jit(ops.GradOperation(get_all=True)(net))(input_x, input_y)
    assert len(grad_res) == 2
