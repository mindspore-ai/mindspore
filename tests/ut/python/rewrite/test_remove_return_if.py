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
import inspect
import ast
import astunparse

from mindspore.rewrite.ast_transformers import RemoveReturnOutOfIf


class TestIf:
    """Simple test."""
    @staticmethod
    def construct(x):
        """construct"""
        if x > 2:
            return x - 2
        return x


class TestIf2:
    """Test multiple if and test if in if."""
    @staticmethod
    def construct(x):
        """construct"""
        if x > 2:
            return x
        x += 2
        if x > 2:
            if x > 2:
                return x
            x += 2
            return x
        x *= 2
        return x


class TestIf3:
    """Test else."""
    @staticmethod
    def construct(x):
        """construct"""
        if x > 2:
            x -= 2
        else:
            return x
        x -= 2
        return x


class TestIf4:
    """Test elif."""
    @staticmethod
    def construct(x):
        """construct"""
        if x > 2:
            return x
        x += 2
        if x > 2:
            x += 1
            if x > 2:
                x *= 2
            elif x > 3:
                x /= 3
            else:
                return x
            x += 2
            return x
        x *= 2
        return x


def test_simple_if():
    """
    Feature: Test remove return from simple if.
    Description: Test remove return from simple if.
    Expectation: Success.
    """
    ast_root: ast.Module = ast.parse(inspect.getsource(TestIf))
    folder = RemoveReturnOutOfIf()
    folder.transform(ast_root)
    assert astunparse.unparse(ast_root) == """\n\nclass TestIf():
    'Simple test.'\n
    @staticmethod
    def construct(x):
        'construct'
        if (x > 2):
            output_0 = (x - 2)
        else:
            output_0 = x
        return output_0
"""


def test_multiple_if():
    """
    Feature: Test remove return from multiple if.
    Description: Test remove return from multiple if.
    Expectation: Success.
    """
    ast_root: ast.Module = ast.parse(inspect.getsource(TestIf2))
    folder = RemoveReturnOutOfIf()
    folder.transform(ast_root)
    assert astunparse.unparse(ast_root) == """\n\nclass TestIf2():
    'Test multiple if and test if in if.'\n
    @staticmethod
    def construct(x):
        'construct'
        if (x > 2):
            output_2 = x
        else:
            x += 2
            if (x > 2):
                if (x > 2):
                    output_0 = x
                else:
                    x += 2
                    output_0 = x
                output_1 = output_0
            else:
                x *= 2
                output_1 = x
            output_2 = output_1
        return output_2
"""


def test_else():
    """
    Feature: Test remove return in else of if node.
    Description: Test remove return in else of if node.
    Expectation: Success.
    """
    ast_root: ast.Module = ast.parse(inspect.getsource(TestIf3))
    folder = RemoveReturnOutOfIf()
    folder.transform(ast_root)
    assert astunparse.unparse(ast_root) == """\n\nclass TestIf3():
    'Test else.'\n
    @staticmethod
    def construct(x):
        'construct'
        if (x > 2):
            x -= 2
            x -= 2
            output_0 = x
        else:
            output_0 = x
        return output_0
"""


def test_elif():
    """
    Feature: Test remove return from elif.
    Description: Test remove return from elif.
    Expectation: Success.
    """
    ast_root: ast.Module = ast.parse(inspect.getsource(TestIf4))
    folder = RemoveReturnOutOfIf()
    folder.transform(ast_root)
    assert astunparse.unparse(ast_root) == """\n\nclass TestIf4():
    'Test elif.'\n
    @staticmethod
    def construct(x):
        'construct'
        if (x > 2):
            output_3 = x
        else:
            x += 2
            if (x > 2):
                x += 1
                if (x > 2):
                    x *= 2
                    x += 2
                    output_1 = x
                else:
                    if (x > 3):
                        x /= 3
                        x += 2
                        output_0 = x
                    else:
                        output_0 = x
                    output_1 = output_0
                output_2 = output_1
            else:
                x *= 2
                output_2 = x
            output_3 = output_2
        return output_3
"""