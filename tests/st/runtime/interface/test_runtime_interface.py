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
# ==============================================================================
"""
test split interface in common api.
"""
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.graph.api import _frontend_compile, _graph_split
from tests.mark_utils import arg_mark

context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split():
    """
    Feature: split interface.
    Description: split graph and run.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        """
        Test net in testcase for split op.
        """
        def __init__(self):
            super().__init__()
            self.assignadd = P.AssignAdd()
            self.add = P.Add().add_prim_attr("split_op", True).add_prim_attr("func_id", "add_func")

        def construct(self, x, y):
            self.assignadd(x, y)
            z1 = x + y
            z2 = z1 + 3
            z3 = self.add(z1, z2)
            z4 = z1 + z3
            return z4

    x = ms.Tensor(2)
    y = ms.Tensor(3)
    net = Net()
    front_graph = _frontend_compile(net)(x, y)
    fragments = _graph_split(front_graph)
    print("fragments:", fragments)
    print("fragment 1:", fragments[1])
    print("fragment 1 id:", fragments[1].id())
    print("fragment 1 is graph:", fragments[1].is_graph())
    print("fragment 1 key:", fragments[1].py_key())
    print("fragment 1 args list:", fragments[1].args_list())

    res0 = fragments[0](x, y)
    res1 = fragments[1](res0[0], res0[1])
    res2 = fragments[2](res0[1], res1[0], res0[0])
    assert res2[0] == 21


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_multi_op():
    """
    Feature: split interface.
    Description: split graph and run.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        """
        Test net in testcase for split op.
        """
        def __init__(self):
            super().__init__()
            self.add = P.Add().add_prim_attr("split_op", True).add_prim_attr("func_id", "add_func")
            self.sub = P.Sub().add_prim_attr("split_op", True).add_prim_attr("func_id", "sub_func")

        def construct(self, x, y):
            z1 = x + y
            z2 = self.add(z1, x)
            z3 = self.sub(z2, x)
            z4 = z1 + z2 + z3
            return z4

    x = ms.Tensor(2)
    y = ms.Tensor(3)
    net = Net()
    front_graph = _frontend_compile(net)(x, y)
    fragments = _graph_split(front_graph)
    print("fragments:", fragments)
    res0 = fragments[0](x, y)
    res1 = fragments[1](res0[0], x)
    res2 = fragments[2](res0[0], res1[0])
    res3 = fragments[3](res1[0], x)
    res4 = fragments[4](res2[0], res3[0])
    assert res4[0] == 17
