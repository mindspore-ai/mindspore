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
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.rewrite.ast_helpers import AstFinder


class SimpleNet(nn.Cell):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.aaa = 1
        self.bbb = F.add(1, 1)

    def construct(self, x):
        x = self.aaa + x
        x = self.bbb + x
        return x


def test_finder_single_type():
    """
    Feature: Class AstFinder in Package rewrite.
    Description: Use AstFinder to find all Assign ast node.
    Expectation: AstFinder can find all Assign ast node.
    """
    ast_root = ast.parse(inspect.getsource(SimpleNet))
    finder = AstFinder(ast_root)
    results = finder.find_all(ast.Assign)
    assert len(results) == 4
    for result in results:
        assert isinstance(result, ast.Assign)


def test_finder_multi_type():
    """
    Feature: Class AstFinder in Package rewrite.
    Description: Use AstFinder to find all Assign and Attribute ast node.
    Expectation: AstFinder can find all Assign and Attribute ast node.
    """
    ast_root = ast.parse(inspect.getsource(SimpleNet))
    finder = AstFinder(ast_root)
    results = finder.find_all((ast.Assign, ast.Attribute))
    assert len(results) == 11
    assign_num = 0
    attribute_num = 0
    for result in results:
        if isinstance(result, ast.Assign):
            assign_num += 1
            continue
        if isinstance(result, ast.Attribute):
            attribute_num += 1
            continue
        assert False
    assert assign_num == 4
    assert attribute_num == 7
