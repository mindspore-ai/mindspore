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
""" Parse ast.For node """
import ast
import astunparse

from mindspore.rewrite.api.scoped_value import ScopedValue, ValueType
from mindspore.rewrite.ast_helpers.ast_modifier import AstModifier
from mindspore import log as logger
from ..symbol_tree import SymbolTree
from ..parser import Parser
from ..parser_register import reg_parser
from ..common.event import Event

EVAL_WHITE_LIST = ("self.", "range(", "zip(", "enumerate(", "reversed(")


class ForParser(Parser):
    """ Class that implements parsing ast.For nodes """

    @staticmethod
    def modify_init_ast(stree, i, obj, iter_var_name):
        """Modify the ast node in init function."""
        target = f"{iter_var_name.strip()}_{str(i)}"
        setattr(stree.get_origin_network(), target, obj)
        stree.get_origin_network().insert_child_to_cell(target, obj)
        AstModifier.insert_assign_to_function(stree.get_init_func_ast(),
                                              targets=[ScopedValue(ValueType.NamingValue, "self", target)],
                                              expr=ScopedValue(ValueType.NamingValue, "", "getattr"),
                                              args=[ScopedValue(ValueType.NamingValue, "", "obj"),
                                                    ScopedValue(ValueType.StringValue, "", target)])

    @staticmethod
    def modify_construct_ast(stree, ast_node, old_name, new_name):
        """Modify the ast node in construct function."""
        node_str: str = astunparse.unparse(ast_node)
        node_str = node_str.replace(old_name, new_name)
        module_node = ast.parse(node_str)
        new_node = module_node.body[0]
        return new_node

    def target(self):
        return ast.For

    def process(self, stree: SymbolTree, node: ast.For):
        """ Process ast.For node """
        if isinstance(node.target, ast.Name):
            targets = node.target.id
        iter_code = astunparse.unparse(node.iter)
        if not iter_code.startswith(EVAL_WHITE_LIST):
            logger.warning(
                f"For MindSpore Rewrtie, illegal iteration condition for For node, it must start with{EVAL_WHITE_LIST}")
            return
        if iter_code.startswith("self"):
            iter_code = iter_code.replace("self", "stree.get_origin_network()")
        try:
            iter_obj = eval(iter_code)
        except (NameError, TypeError) as e:
            _info = f"For MindSpore Rewrtie, when eval '{iter_code}' by using JIT Fallback feature, " \
                         f"an error occurred: {str(e)}"
            logger.warning(_info)
            stree.try_append_python_node(node, node)
            return

        iter_var_name = iter_code.split(".")[-1]
        index = stree.get_ast_root().body.index(node) + 1
        if isinstance(iter_obj, list):
            for i, obj in enumerate(iter_obj):
                ForParser.modify_init_ast(stree, i, obj, iter_var_name)
                for body in node.body:
                    new_func_name = f"self.{iter_var_name.strip()}_{str(i)}".strip()
                    new_node = ForParser.modify_construct_ast(stree, body, targets, new_func_name)
                    stree.get_ast_root().body.insert(index, new_node)
                    index += 1
            if stree.get_ori_cls_name() == "SequentialCell":
                stree.on_change(Event.CodeChangeEvent)
            stree.get_ast_root().body.remove(node)
            return
        if isinstance(iter_obj, range):
            logger.warning("For MindSpore Rewrtie, range not support.")
        elif isinstance(iter_obj, zip):
            logger.warning("For MindSpore Rewrtie, zip not support.")
        elif isinstance(iter_obj, enumerate):
            logger.warning("For MindSpore Rewrtie, enumerate not support.")
        else:
            logger.warning("For MindSpore Rewrtie, not supported type: ", type(iter_obj))
        stree.try_append_python_node(node, node)
        return

g_for_parser = reg_parser(ForParser())
