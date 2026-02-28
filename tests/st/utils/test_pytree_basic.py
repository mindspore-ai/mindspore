# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Basic tests for python-side pytree utilities."""
# pylint: disable=import-error,no-name-in-module

from collections import namedtuple

import numpy as np
import pytest

from mindspore import Tensor
from mindspore.utils._pytree import (  # type: ignore[attr-defined]
    tree_flatten,
    tree_unflatten,
    tree_leaves,
    tree_structure,
    tree_map,
    register_pytree_node_class,
)
from tests.mark_utils import arg_mark


def test_tree_flatten_unflatten_builtin():
    """
    Feature: tree_flatten/tree_unflatten for builtin containers.
    Description: Verify flatten + unflatten roundtrip on dict/list/tuple.
    Expectation: Reconstructed tree equals original.
    """
    tree = {"a": [1, 2], "b": (3, 4)}
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3, 4]
    rebuilt = tree_unflatten(treedef, leaves)
    assert rebuilt == tree


def test_tree_leaves_and_structure():
    """
    Feature: tree_leaves/tree_structure.
    Description: Collect leaves and verify treedef counts.
    Expectation: Leaves order and counts match structure.
    """
    tree = (1, {"x": 2, "y": [3]})
    leaves = tree_leaves(tree)
    assert leaves == [1, 2, 3]
    treedef = tree_structure(tree)
    assert treedef.num_leaves == 3
    assert treedef.num_nodes > treedef.num_leaves  # there are internal nodes


def test_tree_map_with_multiple_trees():
    """
    Feature: tree_map with multiple inputs.
    Description: Map over two trees with identical structure.
    Expectation: Elementwise sum matches expected values.
    """
    tree = {"a": [1, 2], "b": (3, 4)}
    other = {"a": [10, 20], "b": (30, 40)}
    out = tree_map(lambda x, y: x + y, tree, other)
    assert out == {"a": [11, 22], "b": (33, 44)}


def test_namedtuple_roundtrip():
    """
    Feature: Namedtuple handling.
    Description: Flatten/unflatten preserves namedtuple type and data.
    Expectation: Result matches original tree with namedtuple intact.
    """
    Point = namedtuple("Point", ["x", "y"])
    tree = {"p": Point(1, (2, 3))}
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3]
    rebuilt = tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt["p"], Point)
    assert rebuilt == tree


@register_pytree_node_class
class Pair:
    """Simple custom pytree node."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def tree_flatten(self):
        return (self.left, self.right), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


def test_custom_node_registration_roundtrip():
    """
    Feature: Custom pytree node registration.
    Description: Ensure custom class roundtrip via register_pytree_node_class.
    Expectation: Flatten/unflatten restores nested custom nodes.
    """
    tree = Pair(Pair(1, 2), Pair(3, 4))
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3, 4]
    rebuilt = tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, Pair)
    assert isinstance(rebuilt.left, Pair)
    assert rebuilt.left.left == 1
    assert rebuilt.right.right == 4


def test_tree_map_structure_mismatch_raises():
    """
    Feature: tree_map structure validation.
    Description: Mismatched structures should raise TypeError.
    Expectation: TypeError is raised for list vs tuple mismatch.
    """
    with pytest.raises(TypeError):
        tree_map(lambda x, y: x, [1, 2], (1, 2))


def test_tensors_only_leaf_flatten_and_unflatten():
    """
    Feature: tensors_only_leaf flatten/unflatten.
    Description: Only Tensor objects become leaves; scalars stay in structure.
    Expectation: Leaf list contains only tensors; unflatten restores structure.
    """
    tree = {"a": Tensor([1, 2]), "b": (3, {"c": Tensor([4])})}
    leaves, treedef = tree_flatten(tree, tensors_only_leaf=True)
    assert len(leaves) == 2
    assert all(isinstance(t, Tensor) for t in leaves)
    rebuilt = tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt["a"], Tensor)
    np.testing.assert_array_equal(rebuilt["a"].asnumpy(), np.array([1, 2]))
    assert rebuilt["b"][0] == 3
    assert rebuilt["b"][1]["c"].asnumpy().item() == 4


def test_tree_leaves_tensors_only_leaf():
    """
    Feature: tensors_only_leaf in tree_leaves.
    Description: Collect only Tensor leaves when tensors_only_leaf is True.
    Expectation: Leaves list contains only tensors, count matches tensors.
    """
    tree = ({"x": Tensor([5])}, 6, Tensor([7]))
    leaves = tree_leaves(tree, tensors_only_leaf=True)
    assert len(leaves) == 2
    assert all(isinstance(t, Tensor) for t in leaves)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pytree_basic_suite():
    """
    Feature: Pytree aggregated suite.
    Description: Run all basic pytree cases including tensors_only_leaf mode.
    Expectation: All sub-tests pass.
    """
    test_tree_flatten_unflatten_builtin()
    test_tree_leaves_and_structure()
    test_tree_map_with_multiple_trees()
    test_namedtuple_roundtrip()
    test_custom_node_registration_roundtrip()
    test_tree_map_structure_mismatch_raises()
    test_tensors_only_leaf_flatten_and_unflatten()
    test_tree_leaves_tensors_only_leaf()
