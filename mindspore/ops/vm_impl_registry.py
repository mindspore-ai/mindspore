# Copyright 2020 Huawei Technologies Co., Ltd
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

"""
User debug operators registry.

User can define a python implementation of primitive operator and
register it with the registry, this mechanism applied for debugging currently.
"""

from ._register_for_op import Registry

vm_impl_registry = Registry()
"""
Register the python primitive debug implementation function of a primitive operator.

Examples:
    >>> @vm_impl_registry.register(P.Type)
    >>> def vm_impl_dtype(self):
    >>>     def vm_impl(x):
    >>>         return type(x)
    >>>     return vm_impl
"""


def get_vm_impl_fn(prim):
    """
    Gets the virtual implementation function by a primitive object or primitive name.

    Args:
        prim (Union[Primitive, str]): primitive object or name for operator register.

    Returns:
        function, vm function
    """
    out = vm_impl_registry.get(prim, None)
    if out:
        return out(prim)
    return None
