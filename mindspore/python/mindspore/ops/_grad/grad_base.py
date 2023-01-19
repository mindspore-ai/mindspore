# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

from mindspore.ops._register_for_op import Registry
from mindspore.ops.primitive import Primitive
from mindspore.common import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.ops.operations._inner_ops import DynamicBroadcastTo
from mindspore.ops import functional as F
from mindspore.ops.operations import _sequence_ops as seq_op
import mindspore as ms
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


def get_bprop_fn(prim, get_closure=False):
    """get bprop function by primitive obj or prim name for c++"""
    out = bprop_getters.get(prim, None)
    if out is None:
        return bprops.get(prim, None)
    if get_closure:
        return out
    return out(prim)


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

    if isinstance(data, list):
        if F.is_sequence_value_unknown(data):
            data_tensor = seq_op.ListToTensor()(data, ms.int64)
            return True, data_tensor
        return False, data
    if isinstance(data, tuple):
        if F.is_sequence_value_unknown(data):
            data_tensor = seq_op.TupleToTensor()(data, ms.int64)
            return True, data_tensor
        return False, data
    if isinstance(data, int):
        if not F.isconstant(data):
            data_tensor = F.scalar_to_tensor(data, ms.int64)
            return True, data_tensor
        return False, data
    if isinstance(data, float):
        if not F.isconstant(data):
            data_tensor = F.scalar_to_tensor(data, ms.float32)
            return True, data_tensor
        return False, data
    return False, data


def dyn_rank(tensor):
    """get the rank of tensor"""
    out_rank = dyn_shape(dyn_shape(tensor))
    return P.Reshape()(out_rank, ())


def dyn_rank_1d(tensor):
    """get the rank of tensor and return a 1D tensor"""
    tensor_shape = dyn_shape(tensor)
    return dyn_shape(tensor_shape)


def dyn_size(tensor, dtype=mstype.int64):
    """get the size of tensor"""
    shape = dyn_shape(tensor)
    shape = cast(shape, mstype.float32)
    size = P.ReduceProd()(shape)
    size = cast(size, dtype)
    return size


def create_tensor_by_element(ori_tuple, data_type=mstype.int64):
    """create tensor by element"""
    if isinstance(ori_tuple, (tuple, list)):
        new_tuple = []
        for tuple_i in ori_tuple:
            tuple_i = P.Cast()(tuple_i, data_type)
            new_tuple.append(tuple_i)
        return P.Stack(axis=-1)(new_tuple)
    return ori_tuple


def dyn_invert_permutation(perm):
    """get the invert premutation of tensor"""
    indices = P.ExpandDims()(perm, -1)
    end = dyn_size(perm)
    updates = P.Range()(P.Cast()(0, end.dtype), end, P.Cast()(1, end.dtype))
    output = P.ZerosLike()(updates)
    new_perm = P.TensorScatterUpdate()(P.Cast()(output, mstype.float32),
                                       indices, P.Cast()(updates, mstype.float32))
    return P.Cast()(new_perm, mstype.int32)


def dyn_fill(value_type, shape, value):
    """create a Tensor of the specified shape of tensor and fill it with the specified value."""
    value_tensor = P.Cast()(value, value_type)
    return DynamicBroadcastTo()(value_tensor, shape)


def dyn_ones(shape, value_type):
    """creates a tensor filled with value ones."""
    return dyn_fill(value_type, shape, 1)


def sum_grad_reduce_axis(x, rx, keep_dims=False):
    """sum the grad_reduce_axis. when rx is an empty tensor and skip_mode is True, we need skip this reduce"""
    return P.ReduceSum(keep_dims=keep_dims, skip_mode=True)(x, rx)
