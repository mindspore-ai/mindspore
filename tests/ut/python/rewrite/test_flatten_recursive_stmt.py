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
from mindspore.rewrite.ast_transformers.flatten_recursive_stmt import FlattenRecursiveStmt


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
    Feature: Class FlattenRecursiveStmt.
    Description: Apply FlattenRecursiveStmt on a simple network.
    Expectation: Success.
    """
    ast_node = _get_ast()
    frs = FlattenRecursiveStmt()
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
