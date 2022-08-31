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

"""debug_ops vmap impl."""
from __future__ import absolute_import

from mindspore.ops import operations as P
from mindspore.ops.primitive import Primitive
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, _raise_value_error


@vmap_rules_getters.register(P.Print)
def get_print_vmap_rule(prim, axis_size):
    """VmapRule for `Print` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(*args):
        vals = ()
        args_len = len(args)
        for index, val_bdim in enumerate(args, 1):
            # Only the monad tag can not be tuple
            if index == args_len:
                vals = vals + (val_bdim,)
                break
            if not isinstance(val_bdim, tuple):
                _raise_value_error("The received args does not contain axis information in P.Print.")
            else:
                val, dim = val_bdim
                if dim is None:
                    vals = vals + (val,)
                else:
                    vals = vals + ("(", val, ", dim: ", dim, ")")

        out = prim(*vals)
        return out

    return vmap_rule
