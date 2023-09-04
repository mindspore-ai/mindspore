# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""common _jit_fallback_utils."""


_local_value_nodes = {}


def generate_list(key_name, list_value):
    "Generate list object."
    list_obj = get_local_variable(key_name)
    # Need to clear current object, in case the same make_list is called multiple times.
    list_obj.clear()
    list_obj.extend(list_value)
    return list_obj


def get_local_variable(name):
    """Get the local variable according name."""
    return _local_value_nodes.get(name)


def set_local_variable(name, value):
    """Set the local variable with name and value."""
    _local_value_nodes[name] = value


def dict_setitem(dic, key, val):
    """Set an element to dict."""
    dic.__setitem__(key, val)
    return dic


def list_inplace_append(list_obj, target_obj):
    """Inplace append target_obj to list_obj for jit fallback."""
    # When input list is empty list, it will be converted to tuple.
    # This will be removed after empty list problem is solved.
    if isinstance(list_obj, tuple) and not list_obj == 0:
        list_obj = []
    list_obj.append(target_obj)
    return list_obj


def list_inplace_extend(list_obj, target_obj):
    """Inplace extend target_obj to list_obj for jit fallback."""
    # When input list is empty list, it will be converted to tuple.
    # This will be removed after empty list problem is solved.
    if isinstance(list_obj, tuple) and not list_obj == 0:
        list_obj = []
    list_obj.extend(target_obj)
    return list_obj


def list_inplace_insert(list_obj, index, target_obj):
    """Inplace insert target_obj to list_obj at position index for jit fallback."""
    # When input list is empty list, it will be converted to tuple.
    # This will be removed after empty list problem is solved.
    if isinstance(list_obj, tuple) and not list_obj == 0:
        list_obj = []
    list_obj.insert(index, target_obj)
    return list_obj


def list_inplace_pop(list_obj, index):
    """Inplace pop list_obj element at position index for jit fallback."""
    # When input list is empty list, it will be converted to tuple.
    # This will be removed after empty list problem is solved.
    if isinstance(list_obj, tuple) and not list_obj == 0:
        list_obj = []
    list_obj.pop(index)
    return list_obj


def list_inplace_reverse(list_obj):
    """Inplace reverse list_obj for jit fallback."""
    # When input list is empty list, it will be converted to tuple.
    # This will be removed after empty list problem is solved.
    if isinstance(list_obj, tuple) and not list_obj == 0:
        list_obj = []
    list_obj.reverse()
    return list_obj


def list_inplace_clear(list_obj):
    """Inplace clear list_obj for jit fallback."""
    # When input list is empty list, it will be converted to tuple.
    # This will be removed after empty list problem is solved.
    if isinstance(list_obj, tuple) and not list_obj == 0:
        list_obj = []
    list_obj.clear()
    return list_obj


def dict_inplace_setitem(dict_obj, key, target):
    """Inplace dictionary setitem operation for dict_obj."""
    dict_obj[key] = target
    return dict_obj
