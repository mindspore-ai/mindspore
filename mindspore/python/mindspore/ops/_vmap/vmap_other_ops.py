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

"""other_ops vmap impl."""
from __future__ import absolute_import

from mindspore.ops import operations as P
from mindspore.ops.primitive import Primitive
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, get_assign_vmap_rule, _raise_value_error, \
    get_unsupported_dynamic_vmap_rule


@vmap_rules_getters.register("Load")
def get_load_vmap_rule(prim, axis_size):
    """VmapRule for `Load` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(ref_bdim, u_monad):
        var, dim = ref_bdim
        out = prim(var, u_monad)
        return out, dim

    return vmap_rule


@vmap_rules_getters.register(P.Identity)
def get_identity_vmap_rule(prim, axis_size):
    """VmapRule for `Identity` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(ref_bdim):
        var, dim = ref_bdim
        out = prim(var)
        return out, dim

    return vmap_rule


@vmap_rules_getters.register("list_getitem")
@vmap_rules_getters.register("TupleGetItem")
def get_seq_get_item_vmap_rule(prim, axis_size):
    """VmapRule for `list_getitem` or `TupleGetItem` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(inputs_seq, index_bdim):
        index, _ = index_bdim
        out = prim(inputs_seq, index)
        return out

    return vmap_rule


@vmap_rules_getters.register("Switch")
@vmap_rules_getters.register("Partial")
def get_partical_vmap_rule(prim, axis_size):
    """VmapRule for `Partial` and `Switch` operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    def vmap_rule(*args):
        vals = ()
        for val_bdim in args:
            if not isinstance(val_bdim, tuple):
                vals = vals + (val_bdim,)
            else:
                val, dim = val_bdim
                if dim is not None:
                    _raise_value_error("In the scenario where vmap contains control flow, currently only the "
                                       "case of each batch branch with the same processing operations is "
                                       "supported, so that the source axis of args in {} must be None, "
                                       "but got {}.".format(prim_name, dim))
                vals = vals + (val,)

        out = prim(*vals)
        return out

    return vmap_rule


get_assign_vmap_rule = vmap_rules_getters.register(P.Assign)(get_assign_vmap_rule)
get_unsupported_dynamic_vmap_rule = \
    vmap_rules_getters.register(P.StandardLaplace)(get_unsupported_dynamic_vmap_rule)
get_unsupported_dynamic_vmap_rule = vmap_rules_getters.register(P.UniformInt)(get_unsupported_dynamic_vmap_rule)
get_unsupported_dynamic_vmap_rule = vmap_rules_getters.register(P.UniformReal)(get_unsupported_dynamic_vmap_rule)
get_unsupported_dynamic_vmap_rule = vmap_rules_getters.register(P.StandardNormal)(get_unsupported_dynamic_vmap_rule)
get_unsupported_dynamic_vmap_rule = \
    vmap_rules_getters.register(P.RandomGamma)(get_unsupported_dynamic_vmap_rule)
