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
"""Fold if return."""

import ast
import copy
from typing import Any, Union
from enum import Enum

from ..common import error_str


class ReturnType(Enum):
    """
    ValueType represents type of nodes.

    - A `NotReturn` represents the node is not a return node.
    - A `IfNotAllReturn` represents the node is an ast.If node and not all branches of it end with return.
    - A `Return` represents the node is a return node or an ast.If node of which all branches of it end with return.
    """
    NotReturn = 0
    IfNotAllReturn = 1
    Return = 2


class RemoveReturnOutOfIf(ast.NodeTransformer):
    """
    Ast optimizer for removing all returns out of if control flow.

    Example one:
        def func(x):
            if x == 1:
                return x - 1
            x += 1
            return x

    will be optimized to
        def func(x):
            if x == 1:
                output_0 = x - 1
            else:
                x += 1
                output_0 = x
            return output_0

    Example two:
        def func(x):
            if x == 1:
                x = 0
            elif x == 2:
                x = 4
            else:
                return x
            x += 1
            return x + 1

    will be optimized to
        def func(x):
            if x == 1:
                x = 0
                x += 1
                output_1 = x + 1
            else:
                if x == 2:
                    x = 4
                    x += 1
                    output_0 = x + 1
                else:
                    output_0 = x
                output_1 = output_0
            return output_1
    """

    @staticmethod
    def _last_node_is_return(node: Union[ast.Return, ast.If]) -> ReturnType:
        """
        Judge whether input node represents a return node.
        Return a numeric value according to different cases:
            0: Input node is not an ast.Return or input node is an ast.If of which all branches not end with ast.Return;
            1: Input node is an ast.If and not all branches end with ast.Return;
            2: Input node is an ast.Return or input node is an ast.If of which all branches end with ast.Return.
        """
        if not isinstance(node, ast.Return) and not isinstance(node, ast.If):
            return ReturnType.NotReturn
        if isinstance(node, ast.Return):  # last node is ast.Return
            return ReturnType.Return
        # all branches of ast.If not end with ast.Return
        if node.body and RemoveReturnOutOfIf._last_node_is_return(node.body[-1]) == ReturnType.NotReturn \
                and (not node.orelse or RemoveReturnOutOfIf._last_node_is_return(node.orelse[-1]) ==
                     ReturnType.NotReturn):
            return ReturnType.NotReturn
        # all branches of ast.If end with ast.Return
        if node.body and RemoveReturnOutOfIf._last_node_is_return(node.body[-1]) == ReturnType.Return \
                and node.orelse and RemoveReturnOutOfIf._last_node_is_return(node.orelse[-1]) == ReturnType.Return:
            return ReturnType.Return
        # not all branches of ast.If end with ast.Return
        return ReturnType.IfNotAllReturn

    @staticmethod
    def _fold_return(father_node: Union[ast.FunctionDef, ast.If], if_node: ast.If, if_index: int, attr: str):
        """
        Fold following nodes into if node when not all branches of ast.If end with ast.Return.

        Args:
            father_node (Union[ast.FunctionDef, ast.If]): Father node.
            if_node (ast.If): A if node.
            if_index (int): Index of the if node in body or or-else of father node.
            attr (str): Attribute of father node, can be 'body' or 'orelse'.

        Raises:
            RuntimeError: Father node has not input attr.
        """
        if not hasattr(father_node, attr):
            raise RuntimeError(error_str(f"Father node has not input attr '{attr}'", father_node=father_node))
        father_node_attr = getattr(father_node, attr)
        if RemoveReturnOutOfIf._last_node_is_return(if_node) == ReturnType.IfNotAllReturn:
            # nodes should be copied to all branches which not end with return
            if if_node.body and RemoveReturnOutOfIf._last_node_is_return(if_node.body[-1]) != ReturnType.Return:
                for index in range(if_index + 1, len(father_node_attr)):
                    node = copy.deepcopy(father_node_attr[index])
                    if_node.body.append(node)
            if not if_node.orelse or (if_node.orelse and RemoveReturnOutOfIf._last_node_is_return(if_node.orelse[-1])
                                      != ReturnType.Return):
                for index in range(if_index + 1, len(father_node_attr)):
                    node = copy.deepcopy(father_node_attr[index])
                    if_node.orelse.append(node)
            # delete original nodes
            remove_num = len(father_node_attr) - if_index - 1
            for _ in range(remove_num):
                father_node_attr.pop()

    @staticmethod
    def _fold(father_node: Union[ast.FunctionDef, ast.If], attr: str):
        """Fold nodes. Iterate into body and orelse of if node."""
        if not hasattr(father_node, attr) or not getattr(father_node, attr):
            return

        if isinstance(getattr(father_node, attr)[-1], ast.If):
            RemoveReturnOutOfIf._fold(getattr(father_node, attr)[-1], 'body')  # if.body
            RemoveReturnOutOfIf._fold(getattr(father_node, attr)[-1], 'orelse')  # if.orelse

        cur_index = len(getattr(father_node, attr)) - 2  # no following nodes to fold when if node is the last one
        while cur_index >= 0:
            child = getattr(father_node, attr)[cur_index]
            if isinstance(child, ast.If):
                RemoveReturnOutOfIf._fold_return(father_node, child, cur_index, attr)
                RemoveReturnOutOfIf._fold(child, 'body')  # if.body
                RemoveReturnOutOfIf._fold(child, 'orelse')  # if.orelse
            cur_index -= 1

    @staticmethod
    def _get_output_names(output_names: [str]):
        """Generate unique output names."""
        name: str = 'output_{}'.format(len(output_names))
        output_names.append(name)
        return name

    @staticmethod
    def _move_out_return(output_names: [str], father_node: Union[ast.FunctionDef, ast.If], attr: str):
        """
        Move all return node out of if nodes.
        Replace all original return nodes in ast.If with ast.Assign nodes which represent 'output = return value'.
        And add new ast.Return node to the end of father node.

        Args:
            output_names ([str]): All unique output names.
            father_node (Union[ast.FunctionDef, ast.If]): Father node.
            attr (str): Attribute of father nodes, can be 'body' or 'orelse'.

        Raises:
            RuntimeError: After iterative processing body and orelse of if nodes not all end with ast.Return.
        """
        if not hasattr(father_node, attr) or not getattr(father_node, attr):
            return

        last_node = getattr(father_node, attr)[-1]
        if isinstance(last_node, ast.If) and RemoveReturnOutOfIf._last_node_is_return(last_node) == ReturnType.Return:
            # the body or orelse of last if node should be ast.Return or ast.If
            if isinstance(last_node.body[-1], ast.If):
                RemoveReturnOutOfIf._move_out_return(output_names, last_node, 'body')
            if isinstance(last_node.orelse[-1], ast.If):
                RemoveReturnOutOfIf._move_out_return(output_names, last_node, 'orelse')

            # assert body and or-else all end with return
            if not isinstance(last_node.body[-1], ast.Return) or not isinstance(last_node.orelse[-1], ast.Return):
                raise RuntimeError(error_str("Body and orelse of if nodes not all end with ast.Return.",
                                             father_node=last_node))
            output_name = RemoveReturnOutOfIf._get_output_names(output_names)
            # replace body return
            body_new_last_node = ast.Assign(
                targets=[ast.Name(id=output_name, ctx=ast.Store())], value=last_node.body[-1].value)
            last_node.body.pop()
            last_node.body.append(body_new_last_node)
            # replace else return
            else_new_last_node = ast.Assign(
                targets=[ast.Name(id=output_name, ctx=ast.Store())], value=last_node.orelse[-1].value)
            last_node.orelse.pop()
            last_node.orelse.append(else_new_last_node)
            # add new return node
            new_return_node = ast.Return(value=ast.Name(id=output_name, cts=ast.Store()))
            getattr(father_node, attr).append(new_return_node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Iterate construct node and fold following nodes into if node when condition is met."""
        if node.name != "construct":
            return node
        RemoveReturnOutOfIf._fold(node, 'body')
        output_names = []
        RemoveReturnOutOfIf._move_out_return(output_names, node, 'body')
        return node

    def transform(self, ast_root):
        """Transform."""
        ast_root = self.visit(ast_root)
        ast_root = ast.fix_missing_locations(ast_root)
        return ast_root
