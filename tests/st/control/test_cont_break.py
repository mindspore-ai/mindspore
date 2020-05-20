# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_cont_break """
import numpy as np
import pytest

from mindspore import Tensor, Model, context
from mindspore.nn import Cell


def run_test(netclass, count, dev):
    context.set_context(mode=context.GRAPH_MODE, device_target=dev)
    net = netclass()
    model = Model(net)
    for _ in range(count):
        input_np = np.random.randn(2, 3).astype(np.float32)
        input_ms = Tensor(input_np)
        output_np = net.construct(input_np)  # run python
        output_ms = model.predict(input_ms)  # run graph
        np.testing.assert_array_almost_equal(output_np, output_ms.asnumpy(), decimal=3)


class for_loop_with_break(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        for i in range(8):
            if i > 5:
                x *= 3
                break
            x = x * 2
        return x


class for_loop_with_continue(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        for i in range(8):
            if i > 5:
                x *= 3
                continue
            x = x * 2
        return x


class for_loop_with_cont_break(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        for i in range(8):
            if i < 3:
                i *= 2
                continue
            if i > 5:
                x *= 3
                break
            x = x * 2
        return x


class for_nested_loop_with_break(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        for _ in range(3):
            for j in range(5):
                if j > 3:
                    x *= 2
                    break
                x = x * 1.5
        return x


class while_with_break(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        i = 0
        while i < 5:
            if i > 3:
                x *= 2
                break
            x = x * 1.5
            i += 1
        return x


class while_with_continue(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        i = 0
        while i < 5:
            if i > 3:
                x *= 2
                i += 1
                continue
            x = x * 1.5
            i += 1
        return x


class while_for_nested(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        i = 0
        while i < 5:
            if i > 3:
                for j in range(3):
                    if j > 1:
                        break
                    x *= 2
                i += 1
                continue
            x = x * 1.5
            i += 1
        return x


class pass_branch(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        i = 0
        while i < 5:
            if i > 3:
                pass
            else:
                x = x * 1.5
            i += 1
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cont_break():
    count = 20
    dev = 'CPU'
    run_test(for_loop_with_break, count, dev)
    run_test(for_loop_with_continue, count, dev)
    run_test(for_loop_with_cont_break, count, dev)
    run_test(for_nested_loop_with_break, count, dev)
    run_test(while_with_break, count, dev)
    run_test(while_with_continue, count, dev)
    run_test(while_for_nested, count, dev)
    run_test(pass_branch, count, dev)
