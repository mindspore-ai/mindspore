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
"""Test jit_class"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.common import jit
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_class_with_jit_decorator():
    """
    Feature: @jit_class and @jit decorator.
    Description: Test nested jit_class with jit decorated method.
    Expectation: The network returns a list without throwing exception.
    Migrated from: test_parse_issue_scenario_supplement.py::test_train_with_return_list
    """

    @ms.jit_class
    class Inner:
        def __init__(self):
            self.value = ms.Tensor(np.array([1, 2, 3]))

    @ms.jit_class
    class InnerNet:
        def __init__(self):
            self.inner = Inner()

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.inner_net = InnerNet()

        @jit
        def construct(self):
            out = self.inner_net.inner.value
            return [out + 1]

    ms.set_context(mode=ms.GRAPH_MODE)
    net = Net()
    out = net()
    assert isinstance(out, list)
