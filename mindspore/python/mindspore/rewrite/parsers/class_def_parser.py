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
import collections
import sys
import os
import inspect
from typing import Union
import ast
import astunparse
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

    def __init__(self):
        """Constructor"""
        super(ClassDefParser, self).__init__()
        self._cell_namespace = CellNamespace('mindspore.nn')

    # a dict of which key is father class, value is the number of args of father class init
    dict_init_args = collections.defaultdict(int)
    # a denied_function_decorator_list which is registered by user
    denied_function_decorator_list = []

    @staticmethod
    def _process_init_func_ast(init_ast: ast.FunctionDef, father_classes: list, class_name: str):
        """Process init func"""
        ClassDefParser._modify_arguments_of_init_func(init_ast)
        new_bodies = ClassDefParser._create_bodys_of_init_func(father_classes, class_name)
        init_ast.body = new_bodies

    @staticmethod
    def _create_bodys_of_init_func(father_classes: [], class_name: str):
        """Modify bodys of init func."""
        new_bodies = []
        # update father class init in new class
        father_class_init_bodies = ClassDefParser._father_class_init_process(father_classes)
        new_bodies.extend(father_class_init_bodies)
        # copy instance variables into new class
        ast_copy_attr = ast.parse(
            "for key, value in obj.__dict__.items(): setattr(self, key, value)").body[0]
        new_bodies.append(ast_copy_attr)
        return new_bodies

    @staticmethod
    def _father_class_init_process(father_classes: []) -> [ast.AST]:
        """Add father class init ast bodies."""
        father_class_init_bodies = []
        for father_class in father_classes:
            if father_class == "Cell" or ".Cell" in father_class:
                father_class_init_ast = ast.parse("super().__init__()").body[0]
            elif ClassDefParser.dict_init_args[father_class] == 1:
                father_class_init_ast = ast.parse(f"{father_class}.__init__(self)").body[0]
            else:
                father_class_init_ast = ast.parse(f"{father_class}.__init__(self, obj)").body[0]
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
    def _process_class_variables_name(stree: SymbolTree, var_name_ast: ast.Name, class_def: type):
        """Process name type of class variables"""
        # find class variables with type of variable name.
        var_name = ClassDefParser.get_ast_name(var_name_ast.value)
        if not var_name:
            logger.error(f"For MindSpore Rewrite, during process class variables, get variable name from "
                         f"{astunparse.unparse(var_name_ast)} failed")
            return
        # get class source code
        father_class_file = inspect.getfile(class_def)
        if not os.path.exists(father_class_file):
            logger.error(f"For MindSpore Rewrite, during process class variables, class code file "
                         f"{father_class_name}:{father_class_file} not exist")
            return
        try:
            with open(father_class_file, "r", encoding="utf-8") as f:
                source_code = f.read()
        except RuntimeError as err:
            logger.error(f"For MindSpore Rewrite, during process class variables, read source code "
                         f"from file {father_class_name}:{father_class_file} failed: {err}")
            return
        # find definition of variables in class source file.
        ast_module: ast.Module = ast.parse(source_code)
        for body in ast_module.body:
            # variable is defined by assign statement.
            if isinstance(body, ast.Assign):
                for target in body.targets:
                    target_name = ClassDefParser.get_ast_name(target)
                    if target_name and var_name == target_name:
                        stree.get_external_ast().append(body)
                        return
            # variable is type of function.
            if isinstance(body, ast.FunctionDef):
                if var_name == body.name:
                    stree.get_external_ast().append(body)
                    return
        logger.info(f"For MindSpore Rewrite, during process class variables, get source code "
                    f"of class variable {var_name} from {astunparse.unparse(var_name_ast)} failed.")
        return

    @staticmethod
    def _process_class_variables(stree: SymbolTree, class_ast: ast.ClassDef, class_def: type):
        """Process class_variables of class"""
        class_bodies = []
        class_bodies.extend(class_ast.body)
        while class_bodies:
            body = class_bodies.pop()
            if isinstance(body, ast.Assign) and isinstance(body.value, ast.Name):
                ClassDefParser._process_class_variables_name(stree, body, class_def)
            elif isinstance(body, (ast.If, ast.IfExp)):
                # Process bodies in ast.If and ast.IfExp
                class_bodies.extend(body.body)
                class_bodies.extend(body.orelse)

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
    def _need_append_to_ast(stree: SymbolTree, father_class_name: str) -> bool:
        """If the class is imported and only has one init func(self), it does not need to append to symbol stree."""
        import_modules_set = set()
        for module_list in stree.get_import_modules_dict().values():
            import_modules_set = import_modules_set.union(set(module_list))
        if father_class_name in import_modules_set and ClassDefParser.dict_init_args[father_class_name] == 1:
            return False
        return True

    @staticmethod
    def _need_skip_module(stree: SymbolTree, base: ast.AST, module_name: str, cur_class_def: type) -> bool:
        """
            Skip the module according to the priority of class def.
            the priority of class with the same name:
                defined class in the current module
                > explicitly imported class from other modules
                > implicitly imported class from other modules
                e.g. mindspore.nn and mindformers.modules.transformer both have TransformerEncoderLayer
                     import mindspore.nn
                     from mindformers.modules.transformer import TransformerEncoderLayer
                     the priority of mindformers.modules.transformer is higher
        """
        # a dict of which key is module name, value is the imported modules list
        import_modules_dict = stree.get_import_modules_dict()
        # get the current module file path
        net_path = inspect.getfile(cur_class_def)
        curr_file_path = os.path.abspath(net_path)[:-3]
        curr_file_path = curr_file_path.replace('/', '.')
        curr_file_path = curr_file_path.replace('\\', '.')
        # if the module is the current file, no need to skip
        if curr_file_path.endswith(module_name):
            return False
        # In the following example, the current module cannot directly get the father class(a.FatherNet),
        # the imported module aaa should be considered.
        # e.g. from .. import aaa
        #      class Net(a.FatherNet):
        import_name = ""
        if isinstance(base, ast.Attribute):
            import_name = astunparse.unparse(base).split('.')[0]
        # the module is imported explicitly, no need to skip
        if module_name in import_modules_dict and import_name in import_modules_dict[module_name]:
            return False
        # In other situations, need to skip
        return True

    @staticmethod
    def _process_father_classes(stree, node: ast.ClassDef, cur_class_def: type) -> list:
        """Process father class."""
        father_classes = []
        for idx, base in enumerate(node.bases):
            father_class_name = ClassDefParser.get_ast_name(base)
            if not father_class_name:
                continue
            father_classes.append(father_class_name)
            if father_class_name == "Cell":
                continue
            for k, m in sys.modules.items():
                if k in ("_ast", "ast"):
                    continue
                if ClassDefParser._need_skip_module(stree, base, k, cur_class_def):
                    continue
                if hasattr(m, father_class_name):
                    father_class_def = getattr(m, father_class_name)
                    if not inspect.isclass(father_class_def):
                        continue
                    ClassDefParser._process_one_father_class(stree, father_class_def, father_class_name)
                    node.bases[idx] = ast.Name(id=father_class_name, ctx=ast.Load())
                    break
            else:
                logger.error(f"Get instance of father_class {father_class_name} failed during parsing ast.ClassDef.")
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
        # process father class's class variables
        ClassDefParser._process_class_variables(stree, father_class_ast, father_class_def)
        # process father class's __init__ function
        need_add_init_func = ClassDefParser._need_add_init_func(father_class_ast)
        if need_add_init_func:
            ClassDefParser._add_init_func(father_class_ast)
        for body in father_class_ast.body:
            if isinstance(body, ast.FunctionDef) and body.name == "__init__":
                # Add function decorator
                ClassDefParser._func_decorator_process(body)
                # If the class has native init, record the number of init parameters
                # to avoiding errors caused by added init function
                if not need_add_init_func:
                    # record the number of init parameters
                    ClassDefParser.dict_init_args[father_class_name] = len(body.args.args)
                # Modify init func
                if need_add_init_func or len(body.args.args) > 1:
                    ClassDefParser._process_init_func_ast(body, father_classes, father_class_ast.name)
        # save father class's ast into symbol tree
        if ClassDefParser._need_append_to_ast(stree, father_class_name):
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
        Parse init and construct in ast.ClassDef.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.ClassDef]): An ast.ClassDef node.
            node_manager (NodeManager): NodeManager those asts belong to.
        """
        # Update network's class name from xxx to xxxOpt in ast
        replacer = AstReplacer(node)
        replacer.replace_all(stree.get_ori_cls_name(), stree.get_opt_cls_name())

        # process network's class variables
        ClassDefParser._process_class_variables(stree, node, type(stree.get_origin_network()))

        # process network's father classes
        stree.set_class_ast(node)
        cur_class_def = type(stree.get_origin_network())
        father_classes = ClassDefParser._process_father_classes(stree, node, cur_class_def)

        # add __init__ function to network if necessary
        if isinstance(stree.get_origin_network(), Cell) and ClassDefParser._need_add_init_func(node):
            ClassDefParser._add_init_func(node)

        for body in node.body:
            if isinstance(body, ast.FunctionDef):
                ClassDefParser._func_decorator_process(body)
                if body.name == "__init__":
                    stree.set_init_func_ast(body)
                    ClassDefParser._process_init_func_ast(body, father_classes, stree.get_opt_cls_name())
                elif body.name == "construct":
                    stree.set_ast_root(body)
                    parser: Parser = ParserRegister.instance().get_parser(ast.FunctionDef)
                    parser.process(stree, body, stree)
                else:
                    logger.info(
                        "Ignoring ast.FunctionDef in ast.ClassDef except __init__ and construct function: %s",
                        body.name)
            else:
                logger.info("Ignoring unsupported node(%s) in ast.ClassDef.", type(body).__name__)

g_classdef_parser = reg_parser(ClassDefParser())
