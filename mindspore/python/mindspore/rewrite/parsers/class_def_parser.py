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
from typing import Dict
import ast
import sys
from textwrap import dedent
from mindspore import log as logger
from mindspore.nn import Cell
from mindspore._extends.parse.namespace import ModuleNamespace
from . import Parser, ParserRegister, reg_parser
from ..node import NodeManager
from ..symbol_tree import SymbolTree
from ..ast_helpers import AstReplacer, AstConverter
from ..common.error_log import error_str


if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse


class ClassDefParser(Parser):
    """Parse ast.ClassDef which is subclass of Cell to SymbolTree."""

    # List of denied function decorators
    denied_function_decorator_list = ["cell_attr_register"]
    # Entry function of the forward computation process
    entry_functions = ["construct"]
    # Final networks where the paring action stops
    final_networks = [Cell]

    def __init__(self):
        """Constructor"""
        super(ClassDefParser, self).__init__()
        self._cell_namespace = ModuleNamespace('mindspore.nn')

    @staticmethod
    def _save_imports(stree):
        """save imports in module where network is located."""
        origin_net = stree.get_origin_network()
        net_path = inspect.getfile(type(origin_net))
        stree.save_imports_from_file(net_path)

    @staticmethod
    def _process_init_func_ast(init_ast: ast.FunctionDef, class_name_ori: str, class_name_opt: str,
                               is_father_class: bool, father_classes: dict):
        """Process init func"""
        ClassDefParser._modify_arguments_of_init_func(init_ast, is_father_class)
        new_bodies = ClassDefParser._create_bodys_of_init_func(class_name_ori, class_name_opt,
                                                               is_father_class, father_classes)
        init_ast.body = new_bodies

    @staticmethod
    def _create_bodys_of_init_func(class_name_ori: str, class_name_opt: str, is_father_class: bool,
                                   father_classes: dict):
        """Modify bodys of init func."""
        new_bodies = []
        # update father class init in new class
        father_class_init_bodies = ClassDefParser._father_class_init_process(father_classes)
        new_bodies.extend(father_class_init_bodies)
        # copy class variables into new class
        if is_father_class:
            ast_copy_attr = ast.parse(
                f"for key, value in {class_name_ori}.__dict__.items():\n"
                f"    if not key.startswith('__'):\n"
                f"        setattr({class_name_opt}, key, value)").body[0]
            new_bodies.append(ast_copy_attr)
        else:
            ast_copy_attr = ast.parse(
                "for key, value in obj.__dict__.items(): setattr(self, key, value)").body[0]
            new_bodies.append(ast_copy_attr)
        return new_bodies

    @staticmethod
    def _father_class_init_process(father_classes: dict) -> [ast.AST]:
        """Add ast bodies of code: father_class.__init__(self)"""
        father_class_init_bodies = []
        for _, class_name in father_classes.items():
            father_class_init_code = f"{class_name}.__init__(self)"
            father_class_init_ast = ast.parse(father_class_init_code).body[0]
            father_class_init_bodies.append(father_class_init_ast)
        return father_class_init_bodies

    @staticmethod
    def _modify_arguments_of_init_func(ast_init_fn: ast.FunctionDef, is_father_class: bool):
        """Replace init function input parameters to self and global_vars."""
        arg_list = [ast.arg(arg="self", annotation="")]
        if not is_father_class:
            arg_list.append(ast.arg(arg="obj", annotation=""))
        ast_init_fn.args = ast.arguments(args=arg_list, posonlyargs=[], kwonlyargs=[],
                                         kw_defaults=[], defaults=[], vararg=None, kwarg=None)
        ast.fix_missing_locations(ast_init_fn)

    @staticmethod
    def _process_class_variables(stree: SymbolTree, function_defs: list):
        """Process class variables of class, only used in child class."""
        init_func_ast = stree.get_init_func_ast()
        for key, value in stree.get_origin_network().__class__.__dict__.items():
            if key.startswith('__'):
                # ignore inner functions
                continue
            is_staticmethod = isinstance(value, staticmethod)
            if not is_staticmethod and callable(value) and key in function_defs:
                # ignore functions defined by self
                continue
            if is_staticmethod:
                assign_code = f"self.{key} = obj.__class__.{key}"
            else:
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
    def _add_ori_cls_into_father_class(stree: SymbolTree, node: ast.ClassDef, class_type: type):
        """Add original class into node.bases as the father class."""
        class_name = class_type.__name__
        # ignore if origin class already exist in bases
        exist_bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
        if class_name in exist_bases:
            return
        # add name to node.bases
        node.bases.insert(0, ast.Name(id=class_name, ctx=ast.Load()))
        # add import
        module = inspect.getmodule(class_type)
        stree.add_import(module, class_name, node)

    @staticmethod
    def _process_father_classes(stree: SymbolTree, node: ast.ClassDef, class_type: type) -> list:
        """Process father class."""
        father_classes: Dict[int, str] = {}
        for idx, base in enumerate(node.bases):
            father_class_type = class_type.__bases__[idx]
            father_class_name = AstConverter.get_ast_name(base)
            if not father_class_name:
                logger.warning(error_str("Failed to parse base class:", child_node=base))
                continue
            if father_class_type in ClassDefParser.final_networks:
                father_classes[idx] = astunparse.unparse(base).strip()
                continue
            # update father class name
            father_class_name_opt = f"{father_class_name}Opt"
            father_classes[idx] = father_class_name_opt
            ClassDefParser._process_one_father_class(stree, father_class_type, father_class_name, father_class_name_opt)
            node.bases[idx] = ast.Name(id=father_class_name_opt, ctx=ast.Load())
        return father_classes

    @staticmethod
    def _process_one_father_class(stree: SymbolTree, class_type: type, class_name_ori: str, class_name_opt: str):
        """Process one father class"""
        # get father class's ast
        source_code = inspect.getsource(class_type)
        class_ast: ast.ClassDef = ast.parse(dedent(source_code)).body[0]
        # update class name from xxx to xxxOpt
        replacer = AstReplacer(class_ast)
        replacer.replace_all(class_name_ori, class_name_opt)
        # process father class's father classes
        father_classes = ClassDefParser._process_father_classes(stree, class_ast, class_type)
        # process father class's __init__ function
        if ClassDefParser._need_add_init_func(class_ast):
            ClassDefParser._add_init_func(class_ast)
        for body in class_ast.body[:]:
            if isinstance(body, ast.FunctionDef) and body.name == "__init__":
                # Add function decorator
                ClassDefParser._func_decorator_process(body)
                ClassDefParser._process_init_func_ast(body, class_name_ori, class_name_opt, True, father_classes)
            else:
                # Remove other codes, which are copied in __init__ function.
                class_ast.body.remove(body)
        # save father class's ast into symbol tree
        stree.get_father_class_ast()[class_ast] = []
        # save father class's file path and imports into symbol tree
        net_path = inspect.getfile(class_type)
        stree.save_imports_from_file(net_path, class_ast)
        # set origin class as father class
        ClassDefParser._add_ori_cls_into_father_class(stree, class_ast, class_type)


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

        # save imports of current class
        stree.set_class_ast(node)
        ClassDefParser._save_imports(stree)

        # process network's father classes
        cur_class_type = type(stree.get_origin_network())
        father_classes = ClassDefParser._process_father_classes(stree, node, cur_class_type)

        # set origin class as father class
        ClassDefParser._add_ori_cls_into_father_class(stree, node, cur_class_type)

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
                    ClassDefParser._process_init_func_ast(body, stree.get_ori_cls_name(),
                                                          stree.get_opt_cls_name(), False, father_classes)
                elif body.name in ClassDefParser.entry_functions:
                    stree.set_ast_root(body)
                    parser: Parser = ParserRegister.instance().get_parser(ast.FunctionDef)
                    parser.process(stree, body, stree)
                else:
                    logger.debug(
                        "Ignoring ast.FunctionDef in ast.ClassDef except __init__ and construct function: %s",
                        body.name)
            elif isinstance(body, (ast.Assign, ast.If, ast.IfExp)):
                # Remove class variables, which are copied in __init__ function.
                node.body.remove(body)
            elif isinstance(body, ast.Expr) and \
                (isinstance(body.value, ast.Str) or (isinstance(body.value, ast.Constant) and \
                                                     isinstance(body.value.value, str))):
                # delete the comments
                node.body.remove(body)
                continue
            else:
                logger.info("Ignoring unsupported node(%s) in ast.ClassDef.", type(body).__name__)
        # Copy function class variables into new network
        ClassDefParser._process_class_variables(stree, function_defs)

g_classdef_parser = reg_parser(ClassDefParser())
