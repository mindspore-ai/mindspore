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
import ast
import inspect

from mindspore.nn import Cell, Conv2d, BatchNorm2d, ReLU
from mindspore.rewrite.ast_helpers.ast_flattener import AstFlattener
from mindspore.rewrite import SymbolTree
from mindspore.ops import operations as P


class Network(Cell):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(16, 16, 3)
        self.bn = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()

    def construct(self, x):
        x = self.conv(x + 1)
        x = x + 1 * 5 + 4 / 2 + self.bn(x)
        self.relu1(x * 5)
        x = self.relu2(x + 1)
        x = True and x or x
        x = self.relu3(x)
        return x + 3


def _get_ast():
    source = inspect.getsource(Network)
    return ast.parse(source)


def test_flatten():
    """
    Feature: Class AstFlattener.
    Description: Apply AstFlattener on a simple network.
    Expectation: Success.
    """
    ast_node = _get_ast()
    frs = AstFlattener()
    frs.transform(ast_node)
    assert len(ast_node.body) == 1
    ast_class = ast_node.body[0]
    assert isinstance(ast_class, ast.ClassDef)
    assert len(ast_class.body) == 2
    ast_init_func = ast_class.body[0]
    assert isinstance(ast_init_func, ast.FunctionDef)
    assert len(ast_init_func.body) == 6
    ast_construct_func = ast_class.body[1]
    assert isinstance(ast_construct_func, ast.FunctionDef)
    assert len(ast_construct_func.body) == 17


class SubNet(Cell):
    """Sample cell which returns multiple features."""
    def __init__(self):
        """Init."""
        super().__init__()
        self.conv = Conv2d(1, 10, 3)

    def construct(self, x):
        """Construct."""
        c1 = self.conv(x)
        c2 = self.conv(c1)
        c3 = self.conv(c2)
        return c1, c2, c3


class NetMultiTargetsWithTupleTargets(Cell):
    """Test cls for multiple targets."""
    def __init__(self):
        """Init."""
        super().__init__()
        self.subnet = SubNet()
        self.add = P.Add()

    def construct(self, x):
        """Construct."""
        add_var_2, (add_var_1, (add_var,)) = self.subnet(x)
        x = self.add(self.add(add_var_1, add_var), add_var_2)
        return x

def test_multi_targets_with_tuple_targets():
    """
    Feature: Test flatten codes with multi tuple-targets.
    Description: Test flatten codes with multi tuple-targets.
    Expectation: Success.
    """
    net = NetMultiTargetsWithTupleTargets()
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("add_var_3 = self.add(add_var_1, add_var)") == 1
    assert codes.count("x = self.add(add_var_3, add_var_2)") == 1
