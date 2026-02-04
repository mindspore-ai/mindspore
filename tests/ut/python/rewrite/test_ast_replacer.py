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
"""Test mindspore.rewrite.ast_helpers.AstReplacer"""
import ast
import re
import inspect
import sys
if sys.version_info >= (3, 9):
    import ast as astunparse  # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.rewrite.ast_helpers import AstReplacer


class SimpleNet2(nn.Cell):
    def construct(self, x):
        return F.add(x, x)


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        SimpleNet._get_int()
        self.aaa = SimpleNet._get_int()
        self.bbb = SimpleNet._get_int() + 1
        self.ccc = F.add(SimpleNet._get_int(), 1)
        self.ddd = SimpleNet2()

    @staticmethod
    def _get_int():
        return 1

    def construct(self, x):
        SimpleNet._get_int()
        aaa = SimpleNet._get_int()
        bbb = SimpleNet._get_int() + aaa
        ccc = F.add(SimpleNet._get_int(), bbb)
        x = self.ddd(ccc)
        return x


def test_replacer():
    """
    Feature: Class AstReplacer in Package rewrite.
    Description:
        Use AstReplacer to replace all "SimpleNet" symbol to "SimpleNet2" symbol.
        Use AstReplacer to undo all replace.
    Expectation: AstReplacer can replace all "SimpleNet" symbol to "SimpleNet2" symbol and restore original ast node.
    """

    original_code = inspect.getsource(SimpleNet)
    assert len(re.findall("SimpleNet", original_code)) == 10
    assert len(re.findall("SimpleNet2", original_code)) == 1

    ast_root = ast.parse(original_code)
    replacer = AstReplacer(ast_root)
    replacer.replace_all("SimpleNet", "SimpleNet2")
    replaced_code = astunparse.unparse(ast_root)
    assert len(re.findall("SimpleNet", replaced_code)) == 10
    assert len(re.findall("SimpleNet2", replaced_code)) == 10

    replacer.undo_all()
    assert len(re.findall("SimpleNet", original_code)) == 10
    assert len(re.findall("SimpleNet2", original_code)) == 1
