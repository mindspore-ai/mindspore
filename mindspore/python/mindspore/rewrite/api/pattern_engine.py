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
"""PatternEngine for modifying SymbolTree by pattern."""
from collections import OrderedDict
from typing import Tuple, Union, List, Type
import abc

from mindspore.nn import Cell
from mindspore.ops.primitive import Primitive
from mindspore import log as logger
from ..._checkparam import Validator
from .node_type import NodeType
from .node import Node
from .symbol_tree import SymbolTree
from .tree_node_helper import TreeNodeHelper


class PatternNode:
    """
    `PatternNode` is defined as a node while defining pattern.

    Args:
        pattern_node_name (str): Name of current node.
        match_type (Type): A type represents what type would be matched of current node. Default: None.
        inputs (list[PatternNode]): Input nodes of current node. Default: None.
    """

    def __init__(self, pattern_node_name: str, match_type: Type = Type[None], inputs: ['PatternNode'] = None):
        Validator.check_value_type("pattern_node_name", pattern_node_name, [str], "PatternNode")
        if inputs is not None:
            Validator.check_element_type_of_iterable("inputs", inputs, [PatternNode], "PatternNode")
        self._name = pattern_node_name
        self._type = match_type
        if inputs is None:
            self._inputs = []
        else:
            self._inputs = inputs

    @staticmethod
    def from_node(node: Node) -> 'PatternNode':
        """
        Create a `PatternNode` from `node`.

        Args:
            node (Node): Input rewrite node.

        Returns:
            A `PatternNode` created from `node`.

        Raises:
            TypeError: If `node` is not a `Node` instance.
        """
        Validator.check_value_type("node", node, [Node], "PatternNode")
        pattern_node: PatternNode = PatternNode(node.get_targets()[0])
        if node.get_node_type() is NodeType.CallCell:
            pattern_node._type = node.get_instance_type()
        return pattern_node

    @staticmethod
    def create_pattern_from_node(node: Node) -> 'PatternNode':
        """
        Create a Pattern from `node` with its inputs.

        Args:
            node (Node): Input rewrite node.

        Returns:
            A `PatternNode` as root of pattern created from rewrite node.

        Raises:
            TypeError: If `node` is not a `Node` instance.
        """
        Validator.check_value_type("node", node, [Node], "PatternNode")
        pattern_node: PatternNode = PatternNode.from_node(node)
        inputs = []
        for node_input in node.get_inputs():
            inputs.append(PatternNode.create_pattern_from_node(node_input))
        pattern_node._inputs = inputs
        return pattern_node

    @staticmethod
    def create_pattern_from_list(type_list: []) -> 'PatternNode':
        """
        Create a Pattern from a cell type list.

        Args:
            type_list (list[type]): Input cell type list.

        Returns:
            A `PatternNode` as root of pattern created from cell type list.

        Raises:
            TypeError: If `type_list` is not a `list`.
        """
        Validator.check_value_type("type_list", type_list, [list], "PatternNode")
        last_node = None
        for i, cell_type in enumerate(type_list):
            cur_node: PatternNode = PatternNode(str(i) + "-" + str(cell_type), cell_type, [])
            if last_node is not None:
                cur_node._inputs = [last_node]
            else:
                cur_node._inputs = []
            last_node = cur_node
        return last_node

    def add_input(self, node):
        """
        Add an input for current `PatternNode`.

        Args:
            node (PatternNode): Cell type as an input.

        Raises:
            TypeError: If `node` is not a `PatternNode` instance.
        """
        Validator.check_value_type("node", node, [PatternNode], "PatternNode")
        self._inputs.append(node)

    def set_inputs(self, inputs):
        """
        Set inputs for current `PatternNode`.

        Args:
            inputs (list[PatternNode]) : Inputs to be set as inputs of current `PatternNode`.

        Raises:
            TypeError: If `inputs` is not a `list` or input in `inputs` is not `PatternNode` instance.
        """
        Validator.check_element_type_of_iterable("inputs", inputs, [PatternNode], "PatternNode")
        self._inputs = inputs

    def match(self, node: Node) -> bool:
        """
        Check if current `PatternNode` can match with `node`.

        Args:
            node (Node) : A rewrite node to be match.

        Raises:
            TypeError: If `node` is not a `Node` instance.
        """
        Validator.check_value_type("node", node, [Node], "PatternNode")
        return self._type == node.get_instance_type()

    def get_inputs(self):
        """
        Getter of inputs.

        Returns:
            A list of `PatternNode`, the inputs of current node.
        """

        return self._inputs

    def name(self) -> str:
        """
        Getter of PatternNode name.
        """
        return self._name

    def type(self):
        """
        Getter of PatternNode type.
        """
        return self._type


class VarNode(PatternNode):
    """
    VarNode is a subclass of `PatternNode` whose `match` method is always return True.
    """

    def __init__(self):
        super(VarNode, self).__init__("placeholder", Cell, [])

    def match(self, node: Node) -> bool:
        return node is not None and node.get_handler() is not None


class Replacement(abc.ABC):
    """
    Interface of replacement function.

    Examples:
        >>> from mindspore.rewrite import Replacement, Node
        >>> from mindspore.nn import nn
        >>> class BnReplacement(Replacement):
        ...     def build(self, pattern, is_chain_pattern: bool, matched):
        ...         bn_node: Node = matched.get(pattern.name())
        ...         conv = nn.Conv2d(16, 16, 3)
        ...         conv_node = Node.create_call_cell(conv, ['x1'], bn_node.get_args(), bn_node.get_kwargs())
        ...         return [conv_node]
    """
    @abc.abstractmethod
    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        """
        Interface define for creating replacement nodes from matched result.

        Note:
            Return value will be delivered into replace api of `SymbolTree` as argument, return value should follow
            restraint of parameter `new_nodes` of `replace` api if `SymbolTree`. See detail in docstring of `replace`
            api of `SymbolTree`.

        Args:
            pattern (PatternNode): A `PatternNode` represents root node of current pattern.
            is_chain_pattern (bool): A bool indicated if pattern is a chain pattern or a tree pattern.
            matched (OrderedDict): An OrderedDict map from pattern_node name to node represents matched result.

        Returns:
            A list of instance of `Node` as replacement nodes.
        """

        raise NotImplementedError

    def __call__(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        return self.build(pattern, is_chain_pattern, matched)


class PatternEngine:
    """
    `PatternEngine` is defined how to transform a `SymbolTree` by `PattenNode`.

    Args:
        pattern (Union[PatternNode, List]): An instance of `PatternNode` or a cell-type-list to construct `PatternNode`
            as root of a pattern.
        replacement (callable): A callable define how to generate new_node.
    """

    def __init__(self, pattern: Union[PatternNode, List], replacement: Replacement = None):
        Validator.check_value_type("replacement", replacement, [Replacement], "PatternEngine")
        if isinstance(pattern, List):
            for p in pattern:
                if not issubclass(p, (Cell, Primitive)):
                    raise TypeError(f"The type of 'pattern' should be one of [Cell, Primitive]"
                                    f"but got '{p}' with type '{type(p).__name__}'.")
        else:
            Validator.check_value_type("pattern", pattern, [PatternNode], "PatternEngine")
        if isinstance(pattern, PatternNode):
            self._is_chain = False
            self._replacement: Replacement = replacement
            self._pattern: PatternNode = pattern
        elif isinstance(pattern, list):
            self._is_chain = True
            self._replacement: Replacement = replacement
            self._pattern: PatternNode = PatternNode.create_pattern_from_list(pattern)
        else:
            raise RuntimeError("Unsupported pattern define")

    def pattern(self) -> PatternNode:
        """
        Getter of pattern.

        Returns:
            A instance of `PatternNode`, used to indicate the type that the current pattern needs to match.
        """

        return self._pattern

    @staticmethod
    def _multi_to_multi_replace(stree: SymbolTree, old_root: Node, matched_dict: OrderedDict,
                                new_nodes: [Node]) -> Node:
        """
        Replace multi-nodes in `stree` by another list of nodes.

        Note:
            Call replace api of `SymbolTree`, so parameter `new_nodes` has same restraint with parameter `new_nodes` of
            `replace` api if `SymbolTree`. See detail in docstring of `replace` api of `SymbolTree`.

        Args:
            stree (SymbolTree): A `SymbolTree` which replacement will apply on.
            old_root (Node): A `Node` represents root of original nodes.
            matched_dict (OrderedDict): An instance of OrderedDict as match result, where key is the pattern name, value
                is the matched node.
            new_nodes (list[Node]): A list of instance of Node as replacement.
        """
        to_erase_list = matched_dict.values()
        # keep all old nodes' inputs
        inputs_dict = {}
        for node in to_erase_list:
            inputs_dict[node.get_name()] = (node.get_inputs())
        # call replace of SymbolTree
        new_root = stree.replace(old_root, new_nodes)
        # replace only support one-to-one replace or one-to-multi replace, we need to erase nodes except
        # cur_node manually
        queue: [Node] = [old_root]
        while queue:
            cur_node: Node = queue.pop(0)
            if cur_node in to_erase_list:
                if cur_node.get_users():
                    # if cur_node is depended on by other node, skip now.
                    # cur_node will be push into queue and be erased later
                    continue
                if stree.get_node(cur_node.get_name()) is not None:
                    # cur_node is not erased before
                    stree.erase_node(cur_node)
                queue.extend(inputs_dict.get(cur_node.get_name()))
        return new_root

    @staticmethod
    def _multi_replace_cellcontainer(stree, cellcontainer, node, matched_dict, new_nodes):
        """Replace node in CellContainer."""
        to_erase_list = list(matched_dict.values())
        stree.replace(Node(node), new_nodes)
        for n in reversed(to_erase_list):
            if n.get_handler() is node:
                continue
            stree.erase_node(n)

    def apply(self, stree: SymbolTree) -> bool:
        """
        Apply current pattern to a `SymbolTree`.

        Note:
            Sub-tree node will be supported in the near feature.

        Args:
            stree (SymbolTree): A `SymbolTree` to be transformed.

        Returns:
            A bool represents if `stree` been changed.

        Raises:
            TypeError: If `stree` is not a `SymbolTree` instance.
        """
        Validator.check_value_type("stree", stree, [SymbolTree], "PatternEngine")
        changed = False
        # IR match
        queue: [Node] = stree.get_inputs()
        # Why need visited: we don't need or should not to visit same node multi-times because pre-visited node may
        # already been erased from SymbolTree.
        # When will we visit same node multi-times:
        #      a
        #     / \
        #    /   \
        #   b     c
        #   |     |
        #   |     d
        #   \    /
        #    \  /
        #     e
        #  1. Visit e, e does not match pattern, add b, d to queue.
        #  2. Visit b, b does not match pattern, add a to queue.
        #  3. Visit d, d does not match pattern, add c to queue.
        #  4. Visit a, a matches pattern and erased from SymbolTree, add xx to queue.
        #  5. Visit c, d does not match pattern, add a to queue.
        #  At step 5, a is visited at second time but a is erased from SymbolTree at step 4.
        visited: [Node] = []
        while queue:
            cur_node: Node = queue.pop(0)
            if cur_node is None:  # Because inputs of node is allowed to be None in replacement.
                continue
            if cur_node in visited:
                continue
            if cur_node.get_node_type() == NodeType.Tree:
                subtree = TreeNodeHelper.get_sub_tree(cur_node)
                self.apply(subtree)
                visited.append(cur_node)
                queue.extend(cur_node.get_users())
                continue
            if cur_node.get_node_type() == NodeType.CellContainer:
                self._process_cellcontainer(stree, cur_node.get_handler())
                continue
            visited.append(cur_node)
            matched, matched_dict = self._match(self._pattern, cur_node)
            # not matched
            if not matched or not PatternEngine._check_match(self._pattern, matched_dict):
                queue.extend(cur_node.get_users())
                continue
            # matched
            new_nodes: [Node] = []
            if self._replacement is not None:
                new_nodes: [Node] = self._replacement(self._pattern, self._is_chain, matched_dict)
            if not new_nodes:  # if replacement is empty, do nothing
                queue.extend(cur_node.get_users())
            else:  # replace cur_node with new_nodes
                changed = True
                root = PatternEngine._multi_to_multi_replace(stree, cur_node, matched_dict, new_nodes)
                queue.append(root)
        return changed

    @staticmethod
    def _merge_ordered_dict(dict1: OrderedDict, dict2: OrderedDict) -> OrderedDict:
        """
        A static util method to merge two OrderedDict.

        Args:
            dict1 (OrderedDict): First dict to be merged.
            dict2 (OrderedDict): Second dict to be merged.

        Returns:
            Merged OrderedDict.
        """

        merged = dict1.copy()
        merged.update(dict2)
        return merged

    def _match(self, pattern: PatternNode, node: Node) -> Tuple[bool, OrderedDict]:
        """
        Match `pattern` with a `node` with all inputs of the `pattern`.

        Args:
            pattern (PatternNode): Pattern to be match.
            node (Node): Node to be match.

        Returns:
            A bool value to indicate if matched.
            An instance of OrderedDict as match result, where key is the pattern name, value is the matched node.
        """

        # Don't iterate into subgraph node, pattern should not be matched across sub-tree
        if node.get_node_type() not in (NodeType.CallCell, NodeType.CallPrimitive, NodeType.Input):
            logger.debug("Pattern match failed: node(%s) is not a CallCell, CallPrimitive or Input", str(node))
            return False, OrderedDict()
        if not pattern.match(node):
            logger.debug("Pattern match failed: node(%s)'s type is %s while pattern type is %s", str(node),
                         node.get_instance_type(), pattern.type())
            return False, OrderedDict()
        if isinstance(pattern, VarNode):
            return True, OrderedDict()
        pattern_inputs = pattern.get_inputs()
        cur_inputs = node.get_inputs()
        input_num = len(pattern_inputs)
        if input_num == 0:
            return True, OrderedDict({pattern.name(): node})
        if input_num != len(cur_inputs):
            logger.debug("Pattern match failed: node(%s)'s has %d inputs while pattern has %d inputs", str(node),
                         len(node.get_inputs()), input_num)
            return False, OrderedDict()
        result = OrderedDict()
        for i in range(0, input_num):
            is_matched, tmp_result = self._match(pattern_inputs[i], cur_inputs[i])
            if not is_matched:
                return False, OrderedDict()
            result = PatternEngine._merge_ordered_dict(result, tmp_result)
        result[pattern.name()] = node
        return True, result

    @staticmethod
    def _check_match(pattern: PatternNode, match_dict: OrderedDict) -> bool:
        """
        Check if matched result is a leak result.

        A leak result means that the result is matched the `pattern`, but some nodes in result which is not
        corresponding to root of pattern have outputs used by nodes outside of result.

        Args:
            pattern (PatternNode): A `PatternNode` represents pattern to be match.
            match_dict (OrderedDict): A OrderedDict represents matched result.

        Returns:
            A bool value to indicate if matched result leaked.
        """
        matched_nodes = match_dict.values()
        for key in match_dict:
            if key == pattern.name():
                continue
            node: Node = match_dict[key]
            for output in node.get_users():
                if output not in matched_nodes:
                    logger.debug("Check match failed, pattern leaked")
                    return False
        return True

    def _process_cellcontainer(self, stree, cellcontainer):
        """Process CellContainer node."""
        for node in cellcontainer.nodes():
            if node.get_node_type() == NodeType.Tree:
                subtree = node.symbol_tree
                self.apply(SymbolTree(subtree))
                continue
            matched, matched_dict = self._match(self._pattern, Node(node))
            if not matched:
                continue
            new_nodes = []
            if self._replacement is not None:
                new_nodes = self._replacement(self._pattern, self._is_chain, matched_dict)
            if not new_nodes:  # if replacement is empty, do nothing
                continue
            PatternEngine._multi_replace_cellcontainer(stree, cellcontainer, node, matched_dict, new_nodes)
