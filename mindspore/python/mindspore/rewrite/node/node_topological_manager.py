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
"""SymbolTree nodes topological relationship manager."""
from typing import Tuple, List
from mindspore import log as logger
from ..api.scoped_value import ScopedValue
from .node import Node
from ..common.observable import Observable
from ..common.event import Event


class TopoManager(Observable):
    """SymbolTree topological-relationship manager."""

    @staticmethod
    def get_node_users(node: Node) -> List[Tuple[Node, int]]:
        """
        Get all nodes which depend on node.

        Args:
            node (Node): An instance of node.

        Returns:
            A list of nodes represents node users.
        """
        results = []
        for users in node.get_target_users().values():
            if not users:
                continue
            for user in users:
                if user not in results:
                    results.append(user)
        return results

    @staticmethod
    def on_update_target(node: Node, index: int, old_target: ScopedValue, new_target: ScopedValue):
        """
        Update node's dicts while updating target of node.

        Args:
            node (Node): An instance of Node whose target being updated.
            arg_idx (int): An int indicates which target of node being updated.
            old_target (ScopedValue): An instance of ScopedValue represents old target.
            new_target (ScopedValue): An instance of ScopedValue represents new target.
        """
        # Update old_target provider node's target_user dict & old arg's user nodes' arg_providers dict
        old_provider = TopoManager._get_value_provider(node, old_target)
        if old_provider:
            for user in node.get_target_users(index):
                old_provider[0].append_target_users(old_provider[1], user)
                user[0].set_arg_providers(user[1], old_provider)
        else:
            for user in node.get_target_users(index):
                user[0].set_arg_providers(user[1], ())
        # Update new_target node's target_users dict & new user nodes' arg_providers dict
        node.get_target_users().clear()
        provider = TopoManager._get_value_provider(node, new_target)
        if provider:
            TopoManager._update_target_users_by_node(node, index, provider)
        else:
            TopoManager._update_target_users_by_value(node, index, new_target)

    @staticmethod
    def _update_target_users_by_value(node, index, value: ScopedValue):
        """
        Update node's _target_users by ScopedValue when insert a new node.
        This function is called when target is not found in previous nodes, which means a new target name is set.
        """
        search_node = node.get_next()
        while search_node is not None:
            if search_node.get_normalized_args() is not None:
                for arg_index, arg in enumerate(search_node.get_normalized_args().values()):
                    if arg == value:
                        node.append_target_users(index, (search_node, arg_index))
                        search_node.set_arg_providers(arg_index, (node, index))
            if search_node.get_targets() is not None:
                for _, target in enumerate(search_node.get_targets()):
                    if target == value:
                        return
            search_node = search_node.get_next()
        return

    @staticmethod
    def _update_target_users_by_node(node, index, provider: Tuple[Node, int]):
        """
        Update node's _target_users by previous node when insert a new node.
        This function is called when target is found in previous nodes, which means a repeat target name is set.
        """
        # Args of nodes which are between node and provider should not be changed
        # [last provider] -> no change args -> [insert node] -> need change args -> [next provider] -> no change args
        nodes_before_insert = []
        search_node = provider[0].get_next()
        while search_node is not None:
            nodes_before_insert.append(search_node)
            if search_node == node:
                break
            search_node = search_node.get_next()
        provider_target_users = provider[0].get_target_users(provider[1])
        for user in provider_target_users[:]: # copy list by slice to support remove item during iterating
            if user[0] not in nodes_before_insert:
                node.append_target_users(index, user)
                provider_target_users.remove(user)
                user[0].set_arg_providers(user[1], (node, index))

    @staticmethod
    def _get_value_provider(node, value: ScopedValue):
        node = node.get_prev()
        while node is not None:
            if node.get_targets() is not None:
                for index, target in enumerate(node.get_targets()):
                    if target == value:
                        return (node, index)
            node = node.get_prev()
        return ()

    def topo_changed(self):
        """
        The function is executed when an Event.TopologicalChangeEvent event is received.
        """
        self.changed(Event.TopologicalChangeEvent)

    def on_insert_node(self, node: Node):
        """
        Update provider dict and consumer dict while inserting node into SymbolTree and update inputs of node by updated
        provider dict and consumer dict.

        Args:
            node (Node): An instance of Node which been inserted into SymbolTree.
        """
        if node.get_normalized_args() is not None:
            for index, arg in enumerate(node.get_normalized_args().values()):
                provider = TopoManager._get_value_provider(node, arg)
                if provider:
                    node.set_arg_providers(index, provider)
                    provider[0].append_target_users(provider[1], (node, index))
        if node.get_targets() is not None:
            for index, target in enumerate(node.get_targets()):
                provider = TopoManager._get_value_provider(node, target)
                if provider:
                    TopoManager._update_target_users_by_node(node, index, provider)
                else:
                    TopoManager._update_target_users_by_value(node, index, target)
        self.topo_changed()

    def on_erase_node(self, node: Node):
        """
        Update provider dict and consumer dict while erasing node from SymbolTree.

        Args:
            node (Node): An instance of Node which been erased from SymbolTree.
        """
        prev_providers = {}
        # Find previous node with same target of current node.
        for index, target_users in node.get_target_users().items():
            if not target_users:
                continue
            prev_provider = TopoManager._get_value_provider(node, node.get_targets()[index])
            if not prev_provider:
                logger.warning(f"Node {node.get_name()}'s target {index}({node.get_targets()[index]}) is used in node "
                               f"{target_users[0][0].get_name()}'s arg {target_users[0][1]}, "
                               f"no other node provides this target if node {node.get_name()} is erased.")
                prev_providers[index] = None
            else:
                prev_providers[index] = prev_provider
        # Update targets topological of nodes
        for index, prev_provider in prev_providers.items():
            for target_user in node.get_target_users(index):
                if prev_provider is None:
                    target_user[0].get_arg_providers().pop(target_user[1], None)
                else:
                    prev_provider[0].append_target_users(prev_provider[1], target_user)
                    target_user[0].set_arg_providers(target_user[1], prev_provider)
        # Update arguments topological of nodes
        for _, arg_providers in node.get_arg_providers().items():
            if not arg_providers:
                continue
            provider_target_users = arg_providers[0].get_target_users(arg_providers[1])
            for target_user in reversed(provider_target_users):
                if target_user[0] == node:
                    provider_target_users.remove(target_user)
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
        # Update old arg's provider node's target_users.
        old_provider = TopoManager._get_value_provider(node, old_arg)
        if old_provider:
            old_provider_target_users = old_provider[0].get_target_users(old_provider[1])
            for target_user in reversed(old_provider_target_users):
                if target_user[0] == node and target_user[1] == arg_idx:
                    old_provider_target_users.remove(target_user)
                    break
        # Update new arg's provider node's target_users.
        provider = TopoManager._get_value_provider(node, new_arg)
        if provider:
            provider[0].append_target_users(provider[1], (node, arg_idx))
        # Update current node's arg_providers.
        node.set_arg_providers(arg_idx, provider)
        self.topo_changed()

    def on_update_arg_by_node(self, dst_node: Node, arg_idx: int, src_node: Node, out_idx: int):
        """
        Update argument of 'dst_node' by another Node.

        Args:
            dst_node (Node): Node to be modified.
            arg_idx (int): Indicate which input being modified.
            src_node (Node): Node as new input.
            out_idx (int): Indicate which output of 'src_node' as new input of 'dst_node'.
        """
        # Update old arg's provider node's target_users.
        if arg_idx in dst_node.get_arg_providers().keys():
            arg_provider = dst_node.get_arg_providers()[arg_idx]
            if arg_provider:
                provider_target_users = arg_provider[0].get_target_users(arg_provider[1])
                if (dst_node, arg_idx) in provider_target_users:
                    provider_target_users.remove((dst_node, arg_idx))
        # Update new arg's provider node's target_users.
        src_node.append_target_users(out_idx, (dst_node, arg_idx))
        # Update current node's arg_providers.
        dst_node.set_arg_providers(arg_idx, (src_node, out_idx))
        self.topo_changed()
