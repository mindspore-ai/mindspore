# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import mindspore as ms
from mindspore import jit, Tensor
from mindspore.nn import Cell
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_pop():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop():
        x = [1, 2, 3, 4]
        y = [4, 3]
        if isinstance(x.pop(), int):
            return x
        return y

    out = list_pop()
    assert out == [1, 2, 3]


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_pop_tensor():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    class Net(Cell):
        def construct(self, x):
            skips = [x]
            return skips.pop()

    net = Net()
    out = net(Tensor([1], ms.float32))
    assert out == 1
