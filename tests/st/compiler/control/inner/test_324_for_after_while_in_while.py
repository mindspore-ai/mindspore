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

"""Test for after while in while."""

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_for_after_while_in_while():
    """
    Feature: Control flow.
    Description: Nested control flow (for-after-while-in-while) in jit.
    Expectation: Executes correctly and gradients are computable.
    """
    loop_count = 10

    def net(x):
        num = loop_count
        while num > 5:
            x = ops.Add()(x, x)
            num = num - 1
            while num < 2:
                x = ops.Flatten()(x)
        for _ in range(3):
            x = ops.Add()(x, x)
        return x

    input_me = Tensor(np.full((2, 3), 2).astype(np.float32))

    out_graph1 = ms.jit(net)(input_me)
    test_fuc = ops.GradOperation()(net)
    out_graph2 = ms.jit(test_fuc)(input_me)

    out_pynative1 = net(input_me)
    test_fuc = ops.GradOperation()(net)
    out_pynative2 = test_fuc(input_me)
    assert np.allclose(out_graph1.asnumpy(),
                       out_pynative1.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out_graph2.asnumpy(),
                       out_pynative2.asnumpy(), 0.0001, 0.0001)
