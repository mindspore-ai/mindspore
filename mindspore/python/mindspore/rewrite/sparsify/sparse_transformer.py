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
"""Sparsify transformer"""
import ast
import inspect
import textwrap
from collections import deque
import astunparse

from mindspore import ops, nn
from mindspore import log as logger
from mindspore.rewrite.parsers.assign_parser import AssignParser
from mindspore.rewrite.sparsify.utils import ArgType, SparseFunc, sparse_rules, get_sparse_func, builtin_ops, \
    get_binop_name, get_sparse_method_outputs, arg_type_to_prefix_map, get_inputs_outputs


OPS_MODULE = "mindspore.ops."
MAX_RECURSION_DEPTH = 10


def sparsify_helper(f, arg_types, user_defined_rules=None, sparse_name="", full_sparse_rules=None, depth=0):
    """Calls sparse_transformer from raw function."""
    if isinstance(f, nn.Cell):
        tree = ast.parse(textwrap.dedent(inspect.getsource(f.construct)))
        # remove self
        tree.body[0].args.args.pop(0)
        global_vars = f.construct.__globals__
        # pylint: disable=protected-access
        init_vars = f._cells
    else:
        tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        global_vars = f.__globals__
        init_vars = {}
    functiondef = tree.body[0]
    args = [arg.arg for arg in functiondef.args.args]
    type_map = {arg: t for arg, t in zip(args, arg_types)}

    sparse_transformer = SparseTransformer(
        type_map, global_vars, init_vars, user_defined_rules, full_sparse_rules, depth)
    sparse_tree = []
    if not sparse_name:
        sparse_name = functiondef.name
    changed = False
    for body in functiondef.body:
        sparse_body = sparse_transformer.transform(body)
        changed |= sparse_transformer.has_changed()
        sparse_tree.append(sparse_body)
    return_types = sparse_transformer.return_types

    if changed:
        sparse_tree = list(x[0] for x in sparse_transformer.sparse_functiondef.values()) + sparse_tree
        ast_module = ast.Module([ast.FunctionDef(
            sparse_name, functiondef.args, sparse_tree, functiondef.decorator_list, functiondef.returns)])
        return ast_module, True, return_types
    return tree, False, return_types


class SparseTransformer(ast.NodeTransformer):
    """Transformer class for sparsify."""
    def __init__(self, type_map, global_vars, init_vars, user_defined_rules=None, full_sparse_rules=None, depth=0):
        """Init method."""
        super().__init__()
        self.type_map = type_map
        self.global_vars = global_vars
        self.init_vars = init_vars
        self.depth = depth
        self.return_types = (ArgType.NONSPARSE,)
        # maps function name and arg types to sparsified ast and return types, which are then inserted into module
        self.sparse_functiondef = {}
        # maps function name and arg types to return types for ast that do not change after sparsify
        self.origin_functiondef = {}

        # keeps track of arg_type for each operand on the call stack recursively
        self._frames = deque()
        self._changed = False
        # variables for which arg_types diverge with control flow are not supported, and are considered dead
        # after exiting the block
        self._dead_vars = {}
        # full_sparse_rules are inherited from caller cell and takes precedence over generic rules
        if full_sparse_rules:
            self.full_sparse_rules = full_sparse_rules
        else:
            self.full_sparse_rules = {}
            user_defined_rules = user_defined_rules or {}
            self.get_sparse_rules(user_defined_rules)

    @staticmethod
    def make_call(node, name="", args=None):
        """Returns a call node with given name and args, if provided."""
        if name:
            func = ast.Name(name, ast.Load())
        else:
            func = node.func
        if args is None:
            args = node.args
        return ast.Call(func, args, node.keywords)

    def get_sparse_rules(self, user_defined_rules):
        """Generates sparse rules for the transformer from generic sparse rules and user-defined sparse rules."""
        for func, rules in {**sparse_rules, **user_defined_rules}.items():
            for r in rules:
                sparse_func = get_sparse_func(r)
                # sparse rules are accessed by the function object and input arg_types pair
                sparse_func_map = self.full_sparse_rules.get(func, {})
                sparse_func_map[tuple(sparse_func.inputs)] = sparse_func
                self.full_sparse_rules[func] = sparse_func_map

    def transform(self, node):
        """Transforms a single node which represents a stmt in the ast."""
        self.clear_stack()
        self._changed = False
        stmt = self.visit(node)
        return stmt

    def has_changed(self):
        return self._changed

    def add_frame(self):
        self._frames.append([])

    def pop_frame(self):
        return tuple(self._frames.pop())

    def push_onto_frame(self, t):
        if not self._frames:
            raise ValueError("Current frame not initialized!")
        self._frames[-1].append(t)

    def push_all_onto_frame(self, t):
        if not self._frames:
            raise ValueError("Current frame not initialized!")
        for i in t:
            self._frames[-1].append(i)

    def clear_stack(self):
        self._frames.clear()

    def make_sparse_func(self, func, node_type, inputs):
        """Returns SparseFunc by looking up sparse_rules."""
        rules = {}
        if node_type == ast.Call:
            if isinstance(func, nn.Cell):
                func_name = func.__class__.__name__.lower()
            else:
                func_name = getattr(func, "__name__", func)
        elif node_type == ast.BinOp:
            func_name = func
        rules = self.full_sparse_rules.get(func, {})

        if ArgType.ANY in rules:
            sparse_func = rules[ArgType.ANY]
        elif inputs in rules:
            sparse_func = rules[inputs]
        else:
            # attempts to find sparse op based on sparse prefix if sparse rules not found
            sparse_func_name = arg_type_to_prefix_map.get(inputs[0], "$") + "_" + func_name
            sparse_op = getattr(ops, sparse_func_name, None)
            if sparse_op is None:
                if any(input_type != ArgType.NONSPARSE for input_type in inputs):
                    return None
                outputs = (ArgType.NONSPARSE,)
            else:
                func_name = sparse_func_name
                _, outputs = get_inputs_outputs(sparse_op)
            sparse_func = SparseFunc(func_name, inputs, outputs)

        if sparse_func.fn != func:
            self._changed = True
        return sparse_func

    def get_sparse_node(self, node, args, func, arg_types):
        """
        Retrieves target from sparse rules if matches, otherwise sparsify the node by recursively expanding `func`
        until maximum recursion depth is reached. Functions in mindspore.ops are not expanded.
        If no matching sparse rule is found, an error is raised.
        """
        sparse_func = self.make_sparse_func(func, type(node), arg_types)
        if sparse_func is not None:
            if self._changed:
                func_node = ast.Name(sparse_func.fn, ast.Load())
                if sparse_func.fn in self.global_vars:
                    func_node = ast.Name(sparse_func.fn, ast.Load())
                else:
                    func_node = ast.Name("ops", ast.Load())
                    func_node = ast.Attribute(func_node, sparse_func.fn, ast.Load())
                node = ast.Call(func_node, args, node.keywords)
            self.push_all_onto_frame(sparse_func.outputs)
            return node

        if func.__module__[:len(OPS_MODULE)] == OPS_MODULE:
            raise ValueError(f"Sparse rules not registered for {func}!")

        if isinstance(func, nn.Cell):
            class_name = func.__class__.__name__
            func_name = class_name.lower()
            init_args = inspect.getfullargspec(func).args
            if len(init_args) != 1:
                raise ValueError(f"Nested cell {class_name} with arguments for init supported!")
        else:
            func_name = func.__name__
        sparse_func_name = f"sparse_{'_'.join(arg_type_to_prefix_map.get(t, 'default') for t in arg_types)}_{func_name}"
        if (func_name, arg_types) in self.sparse_functiondef:
            self._changed = True
            # pylint: disable=get-dict-value-exception
            self.push_all_onto_frame(self.sparse_functiondef[(func_name, arg_types)][1])
            return SparseTransformer.make_call(node, sparse_func_name, args)
        if (func_name, arg_types) in self.origin_functiondef:
            # pylint: disable=get-dict-value-exception
            self.push_all_onto_frame(self.origin_functiondef[(func_name, arg_types)])
            return node
        if self.depth == MAX_RECURSION_DEPTH:
            raise RuntimeError(f"Maximum recursion depth {MAX_RECURSION_DEPTH} for sparsify reached at {func}!")
        functiondef, changed, return_types = sparsify_helper(
            func, arg_types, sparse_name=sparse_func_name, full_sparse_rules=self.full_sparse_rules,
            depth=self.depth + 1)
        self.push_all_onto_frame(return_types)
        if changed:
            self._changed = True
            self.sparse_functiondef[(func_name, arg_types)] = (functiondef, return_types)
            return SparseTransformer.make_call(node, sparse_func_name, args)
        self.origin_functiondef[(func_name, arg_types)] = return_types
        return SparseTransformer.make_call(node, args=args)

    def map_type_to_target(self, node_target, value_types):
        """Records arg_type for each target."""
        if isinstance(node_target, (ast.Tuple, ast.List)):
            targets = node_target.elts
            if len(targets) != len(value_types):
                raise ValueError(f"Target {astunparse.unparse(node_target)} size and value size not match for "
                                 f"ast.Assign {len(targets)} != {len(value_types)}")
            target_vars = []
            for target in targets:
                if not isinstance(target, ast.Name):
                    raise ValueError(f"Each target {ast.dump(target)} for ast.Assign should be ast.Name!")
                target_vars.append(target.id)
            for var, t in zip(target_vars, value_types):
                self.type_map[var] = t
        elif isinstance(node_target, ast.Name):
            var = node_target.id
            if len(value_types) == 1:
                self.type_map[var] = value_types[0]
            else:
                self.type_map[var] = value_types
        else:
            raise ValueError(f"Targets for ast.Assign not supported for {type(node_target)}!")

    def visit_method(self, node):
        """Visits each node based on node class."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise ValueError(f"{type(node)} is not supported in SparseTransformer!")
        return visitor(node)

    def visit(self, node):
        """Visitor interface for all nodes."""
        if not node._fields:
            return node
        if isinstance(node, (ast.AugAssign, ast.Expr)):
            return self.visit_generic_stmt(node)
        if isinstance(node, (ast.BoolOp, ast.Compare, ast.Subscript)):
            # node always evaluates to non-sparse values
            return self.visit_generic_expr(node)
        if isinstance(node, (ast.Tuple, ast.List, ast.UnaryOp)):
            # node contains multiple expressions but is not composable
            return self.visit_composite_generic_expr(node)
        if isinstance(node, (ast.Attribute, ast.Num, ast.Str)):
            return self.visit_scalar_expr(node)
        if isinstance(node, (ast.Index, ast.Slice)):
            # node forms only a part of an expression and does not exist as standalone expression
            return self.visit_partial_expr(node)
        return self.visit_method(node)

    def visit_generic_stmt(self, node):
        self.add_frame()
        node = self.generic_visit(node)
        self.pop_frame()
        return node

    def visit_scalar_expr(self, node):
        self.push_onto_frame(ArgType.NONSPARSE)
        return node

    def visit_generic_expr(self, node):
        self.add_frame()
        node = self.generic_visit(node)
        self.pop_frame()
        self.push_onto_frame(ArgType.NONSPARSE)
        return node

    def visit_composite_generic_expr(self, node):
        return self.generic_visit(node)

    def visit_partial_expr(self, node):
        return node

    def visit_Assign(self, node):     # pylint: disable=invalid-name
        """Visitor for ast.Assign."""
        self.add_frame()
        value = self.visit(node.value)
        value_types = self.pop_frame()
        for node_target in node.targets:
            self.map_type_to_target(node_target, value_types)
        return ast.Assign(node.targets, value)

    def visit_BinOp(self, node):     # pylint: disable=invalid-name
        """Visitor for ast.Binop."""
        self.add_frame()
        node = self.generic_visit(node)
        arg_types = self.pop_frame()
        if len(arg_types) != 2:
            raise ValueError(f"Binary op {astunparse.unparse(node)} values for arg_type len({arg_types}) != 2")
        func = get_binop_name(node.op)
        if func:
            sparse_func = self.make_sparse_func(func, type(node), arg_types)
            if sparse_func is None:
                raise ValueError(f"Sparse rules not defined for {arg_types[0]} {func} {arg_types[1]}!")
            outputs = sparse_func.outputs
        else:
            outputs = (ArgType.NONSPARSE,)
        self.push_all_onto_frame(outputs)
        return node

    def visit_Call(self, node):     # pylint: disable=invalid-name
        """Visitor for ast.Call."""
        self.add_frame()
        args = []
        for arg in node.args:
            args.append(self.visit(arg))
        arg_types = self.pop_frame()

        if all(t == ArgType.NONSPARSE for t in arg_types):
            # if none of the arguments is sparse, do nothing
            self.push_onto_frame(ArgType.NONSPARSE)
            return node

        # pylint: disable=protected-access
        func_name = AssignParser._get_func_name(node)
        if func_name is None or func_name == "":
            raise RuntimeError(f"Function not exist for {ast.dump(node)}!")
        # pylint: disable=protected-access
        func_scope = AssignParser._get_func_scope(node)

        if not func_scope:
            if func_name in builtin_ops:
                self.push_onto_frame(ArgType.NONSPARSE)
                return node
            if func_name in self.global_vars:
                # external function with sparse arguments are inlined and cached
                func = self.global_vars[func_name]
                return self.get_sparse_node(node, args, func, arg_types)
            raise ValueError(f"Call to undefined {func_name}!")

        if func_scope in self.global_vars:
            namespace = self.global_vars[func_scope]
            func = getattr(namespace, func_name, None)
            if func is None:
                raise ValueError(f"{func_name} not defined in {namespace}!")
            sparse_func = self.make_sparse_func(func, type(node), arg_types)
            return self.get_sparse_node(node, args, func, arg_types)

        if func_scope == "self":
            func = self.init_vars.get(func_name, None)
            if func is None:
                raise ValueError(f"{func_name} not defined in in Cell.__init__!")
            return self.get_sparse_node(node, args, func, arg_types)

        func_scope_type = self.type_map.get(func_scope, None)
        if func_scope_type is not None:
            # tensor methods
            if func_scope_type == ArgType.NONSPARSE:
                outputs = (ArgType.NONSPARSE,)
            else:
                outputs = get_sparse_method_outputs(func_name, func_scope_type)
            self.push_all_onto_frame(outputs)
            return node
        raise ValueError(f"Undefined var {func_scope}!")

    def visit_Name(self, node):     # pylint: disable=invalid-name
        """Visitor for ast.Name."""
        if node.id in self.type_map:
            tensor_type = self.type_map[node.id]
        elif node.id in self.global_vars:
            logger.warning(f"Global variable {node.id} treaded as nonsparse value by default.")
            tensor_type = ArgType.NONSPARSE
        elif node.id in self._dead_vars:
            # pylint: disable=get-dict-value-exception
            raise ValueError(f"Divergent arg_types {self._dead_vars[node.id]} for {node.id} are currently not "
                             f"supported in control flow and the variable is considered dead upon leaving "
                             f"the block")
        else:
            raise ValueError(f"Undefined variable {node.id}!")

        if isinstance(tensor_type, tuple):
            self.push_all_onto_frame(tensor_type)
        else:
            self.push_onto_frame(tensor_type)
        return node

    def visit_Return(self, node):     # pylint: disable=invalid-name
        """Visitor for ast.Return."""
        self.add_frame()
        node = self.generic_visit(node)
        self.return_types = self.pop_frame()
        return node

    def visit_While(self, node):     # pylint: disable=invalid-name
        """
        Visitor for ast.While.
        Variables for which arg_types diverge with control flow are not supported, and as a fallback routine,
        unsupported variables are treated as out-of-scope after leaving the control flow body.
        """
        self.add_frame()
        test = self.visit(node.test)
        self.pop_frame()
        orig_type_map = self.type_map.copy()
        body = list(self.visit(expr) for expr in node.body)
        for var, t in self.type_map.items():
            if var not in orig_type_map:
                # new variables in while body are considered active after the leaving the block
                orig_type_map[var] = t
            elif orig_type_map[var] != t:
                # variables for which arg_types diverge are considered dead after leaving the block
                self._dead_vars[var] = (t, orig_type_map.pop(var))
        self.type_map = orig_type_map
        orelse = list(self.visit(expr) for expr in node.orelse)
        return ast.While(test, body, orelse)
