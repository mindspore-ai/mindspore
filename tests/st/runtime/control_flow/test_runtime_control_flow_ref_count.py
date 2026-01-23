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
"""
Test ref count for control flow.
"""
import mindspore as ms
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
from tests.st.backend.ms_backend.common.backend_graph import BackendGraph
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_switch_inline_in_subgraph():
    """
    Feature: Test switch inline with other call node.
    Description: Test double call the switch inline sub graph to make sure no reuse of the sub graph output.
    Expectation: Run success.
    """
    shape = (3, 4)

    true_graph = BackendGraph()
    # Cell reuse flag cannot be set to sub graph and root graph.
    true_graph.set_cell_reuse()
    true_para_1 = true_graph.add_parameter(ms.float32, shape)
    true_para_2 = true_graph.add_parameter(ms.float32, shape)
    true_sub = true_graph.add_cnode(true_graph.add_valuenode(P.Sub()), true_para_1, true_para_2)
    true_graph.add_return(true_sub)

    false_graph = BackendGraph()
    false_para_1 = false_graph.add_parameter(ms.float32, shape)
    false_para_2 = false_graph.add_parameter(ms.float32, shape)
    false_Add = false_graph.add_cnode(false_graph.add_valuenode(P.Add()), false_para_1, false_para_2)
    false_graph.add_return(false_Add)

    sub_graph = BackendGraph()
    sub_para_1 = sub_graph.add_parameter(ms.bool, ())
    sub_para_2 = sub_graph.add_parameter(ms.float32, shape)
    sub_para_3 = sub_graph.add_parameter(ms.float32, shape)
    true_graph_value_node = sub_graph.add_valuenode(true_graph)
    false_graph_value_node = sub_graph.add_valuenode(false_graph)
    true_partial = sub_graph.add_cnode("Partial", true_graph_value_node, sub_para_2, sub_para_3)
    false_partial = sub_graph.add_cnode("Partial", false_graph_value_node, sub_para_2, sub_para_3)
    switch = sub_graph.add_cnode("Switch", sub_para_1, true_partial, false_partial)
    sub_call = sub_graph.add_cnode(switch)
    sub_graph.add_return(sub_call)

    root_graph = BackendGraph()
    sub_graph_value_node = root_graph.add_valuenode(sub_graph)
    root_para_1 = root_graph.add_parameter(ms.float32, shape)
    root_para_2 = root_graph.add_parameter(ms.float32, shape)
    root_para_3 = root_graph.add_parameter(ms.bool, ())
    call_1 = root_graph.add_cnode(sub_graph_value_node, root_para_3, root_para_1, root_para_2)
    call_2 = root_graph.add_cnode(sub_graph_value_node, root_para_3, root_para_1, call_1)
    call_3 = root_graph.add_cnode(sub_graph_value_node, root_para_3, call_1, call_2)
    root_graph.add_return(call_3)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    z = Tensor(False)
    out = root_graph(x, y, z)
    print(out)
    assert np.allclose(out.asnumpy(), np.ones(shape).astype(np.float32) * 5)
