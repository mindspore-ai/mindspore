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
"""sparsify implementation"""
import os

from mindspore import ops
from mindspore.rewrite import SymbolTree, ScopedValue
from mindspore.rewrite.ast_helpers import AstModifier
from mindspore.rewrite.sparsify.sparse_transformer import SparseTransformer
from mindspore.rewrite.sparsify.utils import SparseFunc, ArgType


op_vars = vars(ops)


def get_user_defined_rules(sparse_rules, global_vars, tree):
    """Register user-defined sparse rules."""
    user_defined_rules = {}

    def register_callable(fn):
        func_name = fn.__name__
        if global_vars.get(func_name, None) is fn:
            init_targets = [ScopedValue.create_naming_value(func_name, "self")]
            AstModifier.append_global_vars_expr_to_init(tree.get_init_func_ast(), init_targets, func_name)
        elif not op_vars.get(func_name, None) is fn:
            raise ValueError(f"{fn} not found in globals or mindspore.ops!")

    for source, targets in sparse_rules.items():
        if not isinstance(targets, (tuple, list)) or isinstance(targets, SparseFunc):
            targets = [targets]
        else:
            targets = list(targets)
        for sparse_func in targets:
            if isinstance(sparse_func, SparseFunc) and callable(sparse_func.fn):
                register_callable(sparse_func.fn)
            elif callable(sparse_func):
                register_callable(sparse_func)
            rule = user_defined_rules.get(source, [])
            rule.append(sparse_func)
            user_defined_rules[source] = rule

    return user_defined_rules


def sparsify_tree(tree, arg_types, sparse_rules, f):
    """Sparsify SymbolTree object."""
    global_vars = f.construct.__globals__
    user_defined_rules = get_user_defined_rules(sparse_rules, global_vars, tree)

    # skip self
    args = [arg.arg for arg in tree.get_ast_root().args.args[1:]]
    if isinstance(arg_types, tuple):
        if len(args) != len(arg_types):
            raise ValueError(f"arg_types should have the same length as function parameters, but "
                             f"{len(arg_types)} != {len(args)}!")
        type_map = dict(zip(args, arg_types))
    elif isinstance(arg_types, dict):
        if all(isinstance(i, int) for i in arg_types.keys()):
            type_map = {args[i]: arg_types[i] if i in arg_types else ArgType.NONSPARSE for i in range(len(args))}
        elif all(isinstance(i, str) for i in arg_types.keys()):
            type_map = {arg: arg_types[arg] if arg in arg_types else ArgType.NONSPARSE for arg in args}
        else:
            raise ValueError(f"Keys for arg_types {list(arg_types.keys())} should be all ints or all strings!")
    else:
        raise ValueError(f"Unsupported type for arg_types {type(arg_types)}!")

    # pylint: disable=protected-access
    init_vars = f._cells
    sparse_transformer = SparseTransformer(type_map, global_vars, init_vars, user_defined_rules)
    for i, node_ast in enumerate(tree.get_ast_root().body):
        sp_ast = sparse_transformer.transform(node_ast)
        if sparse_transformer.has_changed():
            tree.get_ast_root().body[i] = sp_ast
    for module, _ in sparse_transformer.sparse_functiondef.values():
        tree.get_module_ast().body.append(module)


def sparsify(f, arg_types, sparse_rules=None):
    """
    Sparsify a Cell object by inferring the appropriate sparse function calls to replace the original function calls by
    propagating sparse properties provided in `arg_types`.

    .. warning::
        This is a set of experimental APIs that is subject to change or deletion.

    Args:
        f (Cell): Cell object to be sparsified.
        arg_types (Tuple[ArgType] | Dict[int, ArgType]): The type of argument (sparse csr, sparse coo,
            non-sparse etc.) expected by `f`. If `arg_type` is a tuple, its length should be the same as the number of
            arguments for `f`; if `arg_type` is a dictionary, each key represents an index into the arguments, and
            arguments not referenced by the dictionary are considered to be non-sparse.
        sparse_rules (Dict[str, SparseFunc], optional): Additional sparse rules. Default: ``None`` .
    """
    os.environ["STREE_PYTHON_FALLBACK"] = "1"
    tree = SymbolTree.create(f)
    handler = tree.get_handler()
    sparse_rules = sparse_rules or {}
    sparsify_tree(handler, arg_types, sparse_rules, f)
    os.unsetenv("STREE_PYTHON_FALLBACK")
    return tree.get_network()
