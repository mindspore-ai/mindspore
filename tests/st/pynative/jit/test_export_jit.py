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

import numpy as np
import mindspore as ms
from mindspore import jit
from mindspore import Tensor, export, load, context
from mindspore.nn import GraphCell
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_add_tensor():
    """
    Feature: Test MindIR Export jit with decorator.
    Description: test jit export as mindir.
    Expectation: No exception, assert True.
    """

    @jit
    def add_tensor(i):
        a = Tensor([9, 8, 5], ms.int32)
        return a + i

    context.set_context(mode=context.GRAPH_MODE)
    in_data = Tensor([2, 1, 1], ms.int32)
    expected_out = add_tensor(in_data)
    export(add_tensor, in_data, file_name="tt.mindir", file_format="MINDIR")
    c_graph = load("tt.mindir")
    c_net = GraphCell(c_graph)
    actual_out = c_net(in_data)
    assert np.array_equal(actual_out.asnumpy(), expected_out.asnumpy())
