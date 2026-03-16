# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Lightweight pytree utilities for MindSpore.

This module provides pytree APIs inspired by JAX's ``tree_util``, 
and we add tensors_only_leaf feature to support tensors only leaf node,
and supports complex input types (tuple, dict, etc.).
The core implementation is in C++ for performance.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable  # pylint: disable=wrong-import-position
from typing import Any

from mindspore._c_expression import (
    PyTreeDef,
    _tree_flatten,
    _tree_unflatten,
    _tree_leaves,
    _tree_structure,
    _tree_map,
    _register_pytree_node,
)


def tree_flatten(tree: Any, is_leaf: Callable[[Any], bool] | None = None, tensors_only_leaf: bool = False):
    """Python wrapper calling C++ implementation."""
    return _tree_flatten(tree, is_leaf=is_leaf, tensors_only_leaf=tensors_only_leaf)


def tree_unflatten(treedef: Any, leaves: Iterable[Any]):
    """Rebuild a pytree from treedef and leaves."""
    return _tree_unflatten(treedef, leaves)


def tree_leaves(tree: Any, is_leaf: Callable[[Any], bool] | None = None, tensors_only_leaf: bool = False):
    """Return leaves of a pytree."""
    return _tree_leaves(tree, is_leaf=is_leaf, tensors_only_leaf=tensors_only_leaf)


def tree_structure(tree: Any, is_leaf: Callable[[Any], bool] | None = None, tensors_only_leaf: bool = False):
    """Return PyTreeDef structure of a pytree."""
    return _tree_structure(tree, is_leaf=is_leaf, tensors_only_leaf=tensors_only_leaf)


def tree_map(func: Callable[..., Any], tree: Any, *rest: Any, is_leaf: Callable[[Any], bool] | None = None,
             tensors_only_leaf: bool = False):
    """Map a function over leaves of one or more pytrees."""
    return _tree_map(func, tree, *rest, is_leaf=is_leaf, tensors_only_leaf=tensors_only_leaf)


def register_pytree_node(node_type: type[Any], flatten_func: Callable[[Any], tuple],
                         unflatten_func: Callable[[Any, Any], Any]):
    """Register a custom pytree node type."""
    return _register_pytree_node(node_type, flatten_func, unflatten_func)


def register_pytree_node_class(cls: type[Any]) -> type[Any]:
    """
    Class decorator variant of ``register_pytree_node``.

    The class must implement ``tree_flatten`` and ``tree_unflatten`` methods:
    - tree_flatten(self) -> (children, aux_data)
    - tree_unflatten(aux_data, children) -> instance (classmethod or staticmethod)

    """
    register_pytree_node(cls, getattr(cls, "tree_flatten"), getattr(cls, "tree_unflatten"))  # pylint: disable=function-ret
    return cls


__all__ = [
    "PyTreeDef",
    "tree_flatten",
    "tree_unflatten",
    "tree_leaves",
    "tree_structure",
    "tree_map",
    "register_pytree_node",
    "register_pytree_node_class",
]
