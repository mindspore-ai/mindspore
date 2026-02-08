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

"""Test for in if in while."""

import mindspore as ms
from mindspore import Tensor, ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_for_in_if_in_while():
    """
    Feature: Control flow.
    Description: Nested control flow (for-in-if-in-while) in jit.
    Expectation: Executes correctly and gradients are computable.
    """

    i = Tensor(16, ms.int32)

    def net(x, y):
        while y != i:
            if x > y:
                for _ in range(2):
                    y *= 2
                    break
            x = x * 2
            y = y * 2
            continue
        return x, y

    input_x = Tensor(4, ms.int32)
    input_y = Tensor(2, ms.int32)

    ret_x, ret_y = ms.jit(net)(input_x, input_y)
    assert ret_x == Tensor(16, ms.int32)
    assert ret_y == Tensor(16, ms.int32)
    grad_res = ms.jit(ops.GradOperation(get_all=True)(net))(input_x, input_y)
    assert len(grad_res) == 2
