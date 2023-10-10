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
"""Define the namespace of MindSpore op definition."""
from .._extends.parse.namespace import CellNamespace


_ms_common_ns = CellNamespace('mindspore.common')
_ms_nn_ns = CellNamespace('mindspore.nn')
_ms_ops_ns = CellNamespace('mindspore.ops.operations')
_ms_functional_ns = CellNamespace('mindspore.ops.functional')

# Elements in _subtree_black_list will not be converted to symbol tree.
# Only str and types are stored in _subtree_black_list.
_subtree_black_list = ["QuantizeWrapperCell",]

def is_subtree(cls_inst):
    """Determine whether 'cls_inst' is a subtree."""
    cls_name = type(cls_inst).__name__
    black_list_types = tuple([elem for elem in _subtree_black_list if not isinstance(elem, str)])
    if cls_name in _subtree_black_list or isinstance(cls_inst, black_list_types):
        return False
    if cls_name in _ms_common_ns and isinstance(cls_inst, _ms_common_ns[cls_name]):
        return False
    if cls_name in _ms_nn_ns and isinstance(cls_inst, _ms_nn_ns[cls_name]):
        return False
    if cls_name in _ms_ops_ns and isinstance(cls_inst, _ms_ops_ns[cls_name]):
        return False

    return True


def is_functional(func_name):
    """Determine whether 'cls_name' is a functional."""
    return func_name in _ms_functional_ns


def get_functional(func_name):
    """Get the function corresponding to the func_name."""
    if func_name in _ms_functional_ns:
        return _ms_functional_ns[func_name]
    return None
