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
"""Parse ast.ClassDef which is subclass of Cell to SymbolTree."""
import inspect
from typing import Union, Dict
import ast
from mindspore import log as logger
from mindspore.nn import Cell
from mindspore._extends.parse.namespace import CellNamespace
from ..symbol_tree import SymbolTree
from .parser import Parser
from .parser_register import ParserRegister, reg_parser
from ..ast_helpers import AstReplacer
from ..common import error_str
from ..parsers.module_parser import ModuleParser
from ..node.node_manager import NodeManager


class AstScopeChecker:
    """
    Check scope of ast node meets the constraints recursively.

    Args:
        scope_constraints (str): A string represents constraints of scope.
    """

    def __init__(self, scope_constraints: str):
        self._scope = scope_constraints

    def check(self, node: ast.AST) -> bool:
        """
        Check scope of `node` meets the constraints recursively.

        Args:
            node (ast.AST): A ast.AST node to be checked.

        Returns:
            A bool represents if input `node` meets constraints.
        """
        if isinstance(node, ast.Compare):
            return self._check_compare(node)
        if isinstance(node, ast.Attribute):
            return self._check_attribute(node)
        if isinstance(node, ast.Name):
            return False
        if isinstance(node, ast.BoolOp):
            return self._check_bool(node)
        if isinstance(node, ast.UnaryOp):
            return self._check_unary(node)
        if isinstance(node, ast.Call):
            return self._check_call(node)
        if isinstance(node, (ast.Constant, ast.NameConstant, ast.Bytes, ast.Str, ast.Num)):
            return True
        raise RuntimeError(error_str(f"only support (ast.Compare, ast.Attribute, ast.Name, ast.BoolOp, ast.UnaryOp"
                                     f"ast.Call, ast.Constant, ast.NameConstant, ast.Bytes, ast.Str, ast.Num"
                                     f") as test check, but got ast type '{type(node).__name__}'", father_node=node))

    def _check_attribute(self, node: ast.Attribute):
        """Check an ast.Attribute meets the constraints recursively."""
        if not isinstance(node.value, ast.Name):
            return False
        if node.value.id != self._scope:
            return False
        return True

    def _check_compare(self, node: ast.Compare):
        """Check an ast.Compare meets the constraints recursively."""
        left = node.left
        for comparator in node.comparators:
            if not self.check(comparator):
                return False
        return self.check(left)

    def _check_bool(self, node: ast.BoolOp):
        """Check an ast.BoolOp meets the constraints recursively."""
        for value in node.values:
            if not self.check(value):
                return False
        return True

    def _check_call(self, node: ast.Call):
        """Check an ast.Call meets the constraints recursively."""
        for arg in node.args:
            if not self.check(arg):
                return False
        for kwarg in node.keywords:
            if not self.check(kwarg):
                return False
        return self.check(node.func)

    def _check_unary(self, node: ast.UnaryOp):
        """Check an ast.UnaryOp meets the constraints recursively."""
        return self.check(node.operand)


class ClassDefParser(Parser):
    """Parse ast.ClassDef which is subclass of Cell to SymbolTree."""

    # a denied_function_decorator_list which is registered by user
    denied_function_decorator_list = []
    # Entry function of the forward computation process
    entry_function = "construct"

    def __init__(self):
        """Constructor"""
        super(ClassDefParser, self).__init__()
        self._cell_namespace = CellNamespace('mindspore.nn')

    @staticmethod
    def _process_init_func_ast(init_ast: ast.FunctionDef, class_name: str, is_father_class: bool,
                               father_classes: dict):
        """Process init func"""
        ClassDefParser._modify_arguments_of_init_func(init_ast)
        new_bodies = ClassDefParser._create_bodys_of_init_func(class_name, is_father_class, father_classes)
        init_ast.body = new_bodies

    @staticmethod
    def _create_bodys_of_init_func(class_name: str, is_father_class: bool, father_classes: dict):
        """Modify bodys of init func."""
        new_bodies = []
        # update father class init in new class
        father_class_init_bodies = ClassDefParser._father_class_init_process(father_classes, is_father_class)
        new_bodies.extend(father_class_init_bodies)
        # copy variables into new class
        if is_father_class:
            ast_copy_attr = ast.parse(
                "for key, value in obj.__dict__.items():\n"
                "    if not key.startswith('__'):\n"
                f"        setattr({class_name}, key, value)").body[0]
            new_bodies.append(ast_copy_attr)
        else:
            ast_copy_attr = ast.parse(
                "for key, value in obj.__dict__.items(): setattr(self, key, value)").body[0]
            new_bodies.append(ast_copy_attr)
        return new_bodies

    @staticmethod
    def _father_class_init_process(father_classes: dict, is_father_class: bool) -> [ast.AST]:
        """Add ast bodies of code: father_class.__init__(...)"""
        father_class_init_bodies = []
        for idx, father_class in father_classes.items():
            if father_class == "Cell":
                father_class_init_code = "super().__init__()"
            elif is_father_class:
                father_class_init_code = f"{father_class}.__init__(self, obj.__bases__[{idx}])"
            else:
                father_class_init_code = f"{father_class}.__init__(self, obj.__class__.__bases__[{idx}])"
            father_class_init_ast = ast.parse(father_class_init_code).body[0]
            father_class_init_bodies.append(father_class_init_ast)
        return father_class_init_bodies

    @staticmethod
    def _modify_arguments_of_init_func(ast_init_fn: ast.FunctionDef):
        """Replace init function input parameters to self and global_vars."""
        arg_self = ast.arg(arg="self", annotation="")
        arg_global_vars = ast.arg(arg="obj", annotation="")
        ast_init_fn.args = ast.arguments(args=[arg_self, arg_global_vars], posonlyargs=[], kwonlyargs=[],
                                         kw_defaults=[], defaults=[], vararg=None, kwarg=None)
        ast.fix_missing_locations(ast_init_fn)

    @staticmethod
    def get_ast_name(ast_node: Union[ast.Name, ast.Attribute]) -> str:
        """Get ast id name"""
        if isinstance(ast_node, ast.Name):
            return ast_node.id
        if isinstance(ast_node, ast.Attribute):
            return ast_node.attr
        return ""

    @staticmethod
    def _process_class_variables(stree: SymbolTree, function_defs: list):
        """Process class variables of class, only used in child class."""
        init_func_ast = stree.get_init_func_ast()
        for key, value in stree.get_origin_network().__class__.__dict__.items():
            if key.startswith('__'):
                # ignore inner functions
                continue
            if callable(value) and key in function_defs:
                # ignore functions defined by self
                continue
            assign_code = f"self.__class__.{key} = obj.__class__.{key}"
            assign_ast = ast.parse(assign_code).body[0]
            init_func_ast.body.append(assign_ast)

    @staticmethod
    def _need_add_init_func(cls_ast: ast.ClassDef) -> bool:
        """If the class don't have init func, we need to add an init func"""
        for body in cls_ast.body:
            if isinstance(body, ast.FunctionDef) and body.name == '__init__':
                return False
        return True

    @staticmethod
    def _add_init_func(cls_ast: ast.ClassDef):
        """Add init func with super().__init__()"""
        init_func_ast = ast.parse("def __init__(self): super().__init__()").body[0]
        cls_ast.body.insert(0, init_func_ast)
        ast.fix_missing_locations(cls_ast)

    @staticmethod
    def _process_father_classes(stree, node: ast.ClassDef, cur_class_def: type) -> list:
        """Process father class."""
        father_classes: Dict[int, str] = {}
        for idx, base in enumerate(node.bases):
            father_class_name = ClassDefParser.get_ast_name(base)
            if not father_class_name:
                continue
            father_classes[idx] = father_class_name
            if father_class_name == "Cell":
                continue
            father_class_def = cur_class_def.__bases__[idx]
            ClassDefParser._process_one_father_class(stree, father_class_def, father_class_name)
            node.bases[idx] = ast.Name(id=father_class_name, ctx=ast.Load())
        return father_classes

    @staticmethod
    def _process_one_father_class(stree: SymbolTree, father_class_def: type, father_class_name: str):
        """Process one father class"""
        # save father class's file path and imports into symbol tree
        net_path = inspect.getfile(father_class_def)
        ModuleParser.save_file_path_to_sys(stree, 0, net_path)
        ModuleParser.save_imports_from_file(stree, net_path)
        # get father class's ast
        source_code = inspect.getsource(father_class_def)
        father_class_ast: ast.ClassDef = ast.parse(source_code).body[0]
        # process father class's father classes
        father_classes = ClassDefParser._process_father_classes(stree, father_class_ast, father_class_def)
        # process father class's __init__ function
        if ClassDefParser._need_add_init_func(father_class_ast):
            ClassDefParser._add_init_func(father_class_ast)
        for body in father_class_ast.body[:]:
            if isinstance(body, ast.FunctionDef) and body.name == "__init__":
                # Add function decorator
                ClassDefParser._func_decorator_process(body)
                ClassDefParser._process_init_func_ast(body, father_class_name, True, father_classes)
            else:
                # Remove other codes, which are copied in __init__ function.
                father_class_ast.body.remove(body)
        # save father class's ast into symbol tree
        stree.get_father_class_ast().append(father_class_ast)

    @staticmethod
    def _func_decorator_process(node: ast.FunctionDef):
        """
        User should set the denied function decorators,
        because the symbol_tree cant pass the correct parameters to decorators but the instance "obj".
        """
        for decorator in node.decorator_list[:]:
            decorator_name = ""
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name):
                    decorator_name = func.id
            elif isinstance(decorator, ast.Name):
                decorator_name = decorator.id
            if decorator_name in ClassDefParser.denied_function_decorator_list:
                node.decorator_list.remove(decorator)

    def target(self):
        """Parse target type"""
        return ast.ClassDef

    def process(self, stree: SymbolTree, node: ast.ClassDef, node_manager: NodeManager):
        """
        Parse init and entry function(default: construct) in ast.ClassDef.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.ClassDef]): An ast.ClassDef node.
            node_manager (NodeManager): NodeManager those asts belong to.
        """
        # Update network's class name from xxx to xxxOpt in ast
        replacer = AstReplacer(node)
        replacer.replace_all(stree.get_ori_cls_name(), stree.get_opt_cls_name())

        # process network's father classes
        stree.set_class_ast(node)
        cur_class_def = type(stree.get_origin_network())
        father_classes = ClassDefParser._process_father_classes(stree, node, cur_class_def)

        # add __init__ function to network if necessary
        if isinstance(stree.get_origin_network(), Cell) and ClassDefParser._need_add_init_func(node):
            ClassDefParser._add_init_func(node)

        # save function defs in ast node to filter function class variables.
        function_defs = []
        for body in node.body[:]:
            if isinstance(body, ast.FunctionDef):
                function_defs.append(body.name)
                ClassDefParser._func_decorator_process(body)
                if body.name == "__init__":
                    stree.set_init_func_ast(body)
                    ClassDefParser._process_init_func_ast(body, stree.get_opt_cls_name(), False, father_classes)
                elif body.name == ClassDefParser.entry_function:
                    stree.set_ast_root(body)
                    parser: Parser = ParserRegister.instance().get_parser(ast.FunctionDef)
                    parser.process(stree, body, stree)
                else:
                    logger.info(
                        "Ignoring ast.FunctionDef in ast.ClassDef except __init__ and construct function: %s",
                        body.name)
            elif isinstance(body, (ast.Assign, ast.If, ast.IfExp)):
                # Remove class variables, which are copied in __init__ function.
                node.body.remove(body)
            else:
                logger.info("Ignoring unsupported node(%s) in ast.ClassDef.", type(body).__name__)
        # Copy function class variables into new network
        ClassDefParser._process_class_variables(stree, function_defs)

g_classdef_parser = reg_parser(ClassDefParser())
