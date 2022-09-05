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

"""grad base functions"""

from .._register_for_op import Registry
from ..primitive import Primitive
from ...common import Tensor
from .. import operations as P
from ...common import dtype as mstype

dyn_shape = P.TensorShape()
cast = P.Cast()


class BpropRegistry(Registry):
    """Registry class for registry functions for grad on Primitive or string."""

    def register(self, prim):
        """register the function."""

        def deco(fn):
            """Decorate the function."""
            if isinstance(prim, str):
                self[prim] = fn
            elif issubclass(prim, Primitive):
                self[id(prim)] = fn
                self[prim.__name__] = fn
            return fn

        return deco


class TaylorFpropRegistry(Registry):
    """Registry class for registry functions for taylor grad on Primitive or string."""

    def register(self, prim):
        """register the function."""

        def deco(fn):
            """Decorate the function."""
            if isinstance(prim, str):
                self[prim] = fn
            elif issubclass(prim, Primitive):
                self[id(prim)] = fn
                self[prim.__name__] = fn
            return fn

        return deco


bprop_getters = BpropRegistry()
bprops = BpropRegistry()
taylor_fprop_getters = TaylorFpropRegistry()
taylor_fprops = TaylorFpropRegistry()


def get_bprop_fn(prim):
    """get bprop function by primitive obj or prim name for c++"""
    out = bprop_getters.get(prim, None)
    if out:
        return out(prim)
    return bprops.get(prim, None)


def get_taylor_fprop_fn(prim):
    """get taylor function by primitive obj or prim name for c++"""
    out = taylor_fprop_getters.get(prim, None)
    if out:
        return out(prim)
    return taylor_fprops.get(prim, None)


def convert_to_tensor(data):
    """convert mutable data to tensor"""
    if isinstance(data, Tensor):
        return True, data

    return False, data


def dyn_rank(tensor):
    """get the rank of tensor"""
    return dyn_shape(dyn_shape(tensor))[0]


def dyn_size(tensor):
    """get the size of tensor"""
    shape = dyn_shape(tensor)
    shape = cast(shape, mstype.float32)
    size = P.ReduceProd()(shape)
    size = cast(size, mstype.int32)
    return size
