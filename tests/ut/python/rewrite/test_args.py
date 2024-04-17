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
import mindspore.nn as nn
from mindspore.rewrite import SymbolTree, Node, ScopedValue


class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x

class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sub_net = SubNet()

    def construct(self, x):
        x = self.relu(x)
        x = self.sub_net(x)
        return x

def test_add_arg_to_subtree():
    """
    Feature: Rewrite add args to subtree.
    Description: Rewrite add args to subtree.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTree.create(net)

    sub_net = stree.get_node("sub_net")
    subtree: SymbolTree = sub_net.get_sub_tree()
    # add input for stree
    new_input = Node.create_input("new_input")
    last_input = subtree.get_inputs()[-1]
    subtree.insert(subtree.after(last_input), new_input)
    # add arg for tree node
    sub_net.get_handler().append_kwarg({"new_input": ScopedValue.create_naming_value("x")})
    # check results
    codes = stree.get_code()
    assert codes.count("def construct(self, x, new_input=None):") == 1
    assert codes.count("x = self.sub_net(x, new_input=x)") == 1
