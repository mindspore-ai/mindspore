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
"""SymbolTree topological-relationship manager."""
from typing import Tuple

from .api.scoped_value import ScopedValue
from .node import Node
from .common.observable import Observable
from .common.event import Event


class TopoManager(Observable):
    """SymbolTree topological-relationship manager."""

    def __init__(self):
        """
        Constructor of TopoManager.
        Init provider and consumer.
        Key of dict is an instance of ScopedValue.
        Value of dict is a tuple whose first is an instance of Node, whose second is an index.
        It means node's index th arg is argument
        """
        super().__init__()
        self._target_provider: {ScopedValue, (Node, int)} = {}
        self._target_consumer: {ScopedValue, [(Node, int)]} = {}

    def get_node_users(self, node: Node) -> [Tuple[Node, int]]:
        """
        Get all nodes which depend on node corresponding to node_or_name.

        Args:
            node (Node): An instance of node.

        Returns:
            A list of nodes represents node users.
        """
        targets = node.get_targets()
        results = []
        for target in targets:
            consumers = self._target_consumer.get(target)
            if consumers is None:
                continue
            results.extend(consumers)
        unique_results = []
        for result in results:
            if result not in unique_results:
                unique_results.append(result)
        return unique_results

    def topo_changed(self):
        """
        The function is executed when an Event.TopologicalChangeEvent event is received.
        """
        self.changed(Event.TopologicalChangeEvent)

    def _add_consumer(self, product: ScopedValue, consumer: Node, index):
        """
        Add a consumer to consumer dict.

        Args:
            product (ScopedValue): An instance of ScopedValue represents product to be consumed.
            consumer (Node): An instance of Node represents consumer.
            index (int): A int represents which input of consumer is the product.
        """
        consumers = self._target_consumer.get(product)
        if consumers is None:
            self._target_consumer[product] = [(consumer, index)]
        else:
            self._target_consumer.get(product).append((consumer, index))

    def _erase_provider(self, product: ScopedValue):
        """
        Erase a provider from provider dict.

        Args:
            product (ScopedValue): An instance of ScopedValue represents product to be erased.
        """
        if self._target_provider.get(product) is not None:
            self._target_provider.pop(product)

    def _erase_consumer(self, product: ScopedValue, consumer: Node):
        """
        Erase a consumer from consumer dict.

        Args:
            product (ScopedValue): An instance of ScopedValue represents product whose consumer would be erased.
            consumer (Node): An instance of Node which would be erased as a consumer.
        """
        consumers = self._target_consumer.get(product)
        if consumers is None:
            return
        for i in range(len(consumers) - 1, -1, -1):
            exist_ele = consumers[i]
            if id(exist_ele[0]) == id(consumer):
                consumers.pop(i)

    def _update_node_inputs(self, node: Node) -> [Node]:
        """
        Update inputs of node by current provider dict and consumer dict.

        Args:
            node (Node): An instance of Node whose inputs will be updated.

        Returns:
            A list of instance of nodes represents inputs of node.
        """
        if node.get_normalized_args() is None:
            node.set_inputs([])
            return []
        inputs = []
        for arg in node.get_normalized_args().values():
            provider = self._target_provider.get(arg)
            # some arg of some node may be self.xxx which is not an output of another node
            if provider is not None:
                inputs.append(provider[0])
        node.set_inputs(inputs)
        return inputs

    def on_insert_node(self, node: Node):
        """
        Update provider dict and consumer dict while inserting node into SymbolTree and update inputs of node by updated
        provider dict and consumer dict.

        Args:
            node (Node): An instance of Node which been inserted into SymbolTree.
        """
        if node.get_targets() is not None:
            for i in range(0, len(node.get_targets())):
                target = node.get_targets()[i]
                if target.value == "_":
                    continue
                if self._target_provider.get(target) is not None:
                    raise RuntimeError("target duplicated:", target)
                self._target_provider[target] = (node, i)
        if node.get_normalized_args() is not None:
            for index, arg in enumerate(node.get_normalized_args().values()):
                self._add_consumer(arg, node, index)
        self._update_node_inputs(node)
        self.topo_changed()

    def on_erase_node(self, node: Node):
        """
        Update provider dict and consumer dict while erasing node from SymbolTree.

        Args:
            node (Node): An instance of Node which been erased from SymbolTree.
        """
        if node.get_targets() is not None:
            for target in node.get_targets():
                consumers = self._target_consumer.get(target)
                if consumers is not None and consumers:
                    raise RuntimeError("Only support erase isolated node: ", node.get_name(), target)
                self._erase_provider(target)
        if node.get_normalized_args() is not None:
            for arg in node.get_normalized_args().values():
                self._erase_consumer(arg, node)
        # clear inputs of node rather than call _update_node_inputs because node is already erase from consumer dict
        node.set_inputs([])
        self.topo_changed()

    def on_update_arg(self, node: Node, arg_idx: int, old_arg: ScopedValue, new_arg: ScopedValue):
        """
        Update provider dict and consumer dict while updating argument of node and update inputs of node by updated
        provider dict and consumer dict.

        Args:
            node (Node): An instance of Node whose arguments being updated.
            arg_idx (int): An int indicates which argument of node being updated.
            old_arg (ScopedValue): An instance of ScopedValue represents original argument.
            new_arg (ScopedValue): An instance of ScopedValue represents new argument.
        """
        self._erase_consumer(old_arg, node)
        self._add_consumer(new_arg, node, arg_idx)
        self._update_node_inputs(node)
        self.topo_changed()

    def dump(self, title=""):
        """
        Dump topological relation.

        Args:
            title (str): A string as a title will be printed before dumping topological relation.
        """
        print(f"{title}------------------------------------------------------------------------------------")
        for k, v in self._target_provider.items():
            print(f"{v[0].get_name()} produces {k.value}")
        for k, v in self._target_consumer.items():
            print(f"{k.value} is consumed by: ")
            for ele in v:
                print(ele[0].get_name())
        print(f"-----------------------------------------------------------------------------------------")
