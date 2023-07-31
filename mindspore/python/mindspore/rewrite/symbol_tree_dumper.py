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
"""SymbolTree dumper."""
import inspect

from .node.node import Node
from .api.node_type import NodeType
from .api.scoped_value import ScopedValue, ValueType


class SymbolTreeDumper:
    """SymbolTree dumper."""

    def __init__(self, symbol_tree):
        """
        Constructor of SymbolTreeDumper.

        Args:
             symbol_tree (SymbolTree): An instance of SymbolTree to be dumped.
        """
        self._symbol_tree = symbol_tree
        self._dump_buffer = ""
        self._dump_key2index = {}

    def _reset(self):
        """Reset SymbolTreeDumper."""
        self._dump_buffer = ""
        self._dump_key2index = {}

    def _dump_global_info(self):
        """Dump global info of SymbolTree."""
        self._dump_buffer += f"#SymbolTree entry    : @construct \n"

    def _dump_inputs(self):
        """Dump inputs of SymbolTree."""
        inputs = self._symbol_tree.get_inputs()
        self._dump_buffer += f"#Inputs num     : {len(inputs)}\n"
        for single_input in inputs:
            targets = single_input.get_targets()
            if len(targets) != 1:
                raise RuntimeError("Only support one output per node now")
            target: ScopedValue = targets[0]
            if target.type != ValueType.NamingValue:
                raise RuntimeError("target.type should equal to ValueType.NamingValue")
            if target.scope != "":
                raise RuntimeError("target.scope should be empty")
            input_arg = target.value
            input_name = f"%input_{input_arg}"
            if input_arg in self._dump_key2index.keys():
                raise RuntimeError("input_arg duplicated: ", input_arg)
            self._dump_key2index[input_arg] = input_name
            self._dump_buffer += f"{input_name}\n"
        self._dump_buffer += f"\n"

    def _dump_nodes(self):
        """Dump nodes of SymbolTree."""
        self._dump_buffer += f"Symbol Tree @construct {{ \n"
        node_no = -1

        node: Node = self._symbol_tree.get_head().get_next()
        while node is not None:
            if node.get_node_type() is NodeType.Output:
                self._dump_buffer += f"  Return(%{node_no}) \n"
                self._dump_buffer += f"      : (null) \n"
                self._dump_buffer += f"      # In file {inspect.getfile(type(self._symbol_tree.get_origin_network()))}"

                node = node.get_next()
                continue

            node_no += 1
            self._dump_key2index[node.get_name()] = f"%{node_no}"

            targets = node.get_targets()
            if not targets:
                targets = [None]
            op_type = node.get_instance_type()
            if hasattr(op_type, "__name__"):
                op_type_name = op_type.__name__
            else:
                if hasattr(type(op_type), "__name__"):
                    op_type_name = type(op_type).__name__
                else:
                    raise RuntimeError("op has no attr __name__")
            self._dump_buffer += f"  %{node_no}({targets[0]}) = {op_type_name}"

            args = node.get_normalized_args().values()
            if args:
                arg_str = f""
                for arg in args:
                    if isinstance(arg, str):
                        arg_name = arg
                    elif isinstance(arg, ScopedValue):
                        arg_name = arg.value
                    else:
                        raise RuntimeError(f"Arg type '{type(arg)} 'of '{arg}' is not supported now")

                    if arg_name in self._dump_key2index.keys():
                        arg_str += f"{self._dump_key2index[arg_name]}, "
                    else:
                        arg_str += f"{arg_name}, "
                self._dump_buffer += f"({arg_str[:-2]})"

            self._dump_buffer += f"{{instance name: {node.get_name()}}}"

            self._dump_buffer += f" attributes {{"
            attrs = node.get_attributes()
            if attrs:
                attrs_str = f""
                for attr in attrs:
                    if not isinstance(attr, str):
                        raise TypeError("attr should be str, got: ", type(attr))
                    attrs_str += f"{attr}: {attrs[attr]}, "
                self._dump_buffer += attrs_str[:-2]
            self._dump_buffer += f"}}\n"

            self._dump_buffer += f"      : (null) -> (null)\n"
            cls_real_path = inspect.getfile(node.get_instance_type()) if node.get_instance() else None
            self._dump_buffer += f"      # In file {cls_real_path}\n"
            self._dump_buffer += f"      # In file {inspect.getfile(type(self._symbol_tree.get_origin_network()))}\n"

            node = node.get_next()
        self._dump_buffer += f"}}\n"

    def dump(self):
        """Dump SymbolTree."""
        self._reset()
        self._dump_global_info()
        self._dump_inputs()
        self._dump_nodes()
        print(self._dump_buffer)
