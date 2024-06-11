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
"""Generate vm_impl function for array ops"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import SumExt
from mindspore._c_expression import typing
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.vm_impl_registry import vm_impl_registry as vm_impl_getters
from .vm_interface import vm


# pylint: disable=unused-argument
@vm_impl_getters.register(P.Assign)
def vm_impl_assign(self):
    """Generate vm_impl function for Assign"""

    def vm_impl(x, value, u=None):
        x.assign_value(value)
        return x

    return vm_impl


@vm_impl_getters.register(P.AssignAdd)
def vm_impl_assignadd(self):
    """Generate vm_impl function for Assign"""

    def vm_impl(x, value, u=None):
        x.assign_value(value)
        return x

    return vm_impl


@vm_impl_getters.register(P.ExpandDims)
def vm_impl_expand_dims(self):
    """Generate vm_impl function for ExpandDims"""

    def vm_impl(x, axis):
        if isinstance(x, float):
            x = Tensor(np.array([x]))
        x = x.asnumpy()
        out = vm.expand_dims(x, axis)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.DType)
def vm_impl_dType(self):
    """Generate vm_impl function for DType"""

    def vm_impl(x):
        # update the src type
        return x.dtype

    return vm_impl


@vm_impl_getters.register(P.Cast)
def vm_impl_cast(self):
    """Generate vm_impl function for Cast"""

    def vm_impl(x, t):
        if isinstance(t, type(mstype.tensor_type)):
            t = t.element_type()
        # update the src type
        x = x.asnumpy()
        out = x.astype(mstype.dtype_to_nptype(typing.type_id_to_type(t)))
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Reshape)
def vm_impl_reshape(self):
    """Generate vm_impl function for Reshape"""

    def vm_impl(x, shp):
        x = x.asnumpy()
        out = vm.reshape(x, shp)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Shape)
def vm_impl_shape(self):
    """Generate vm_impl function for Shape"""

    def vm_impl(x):
        shp = vm.shape(x.asnumpy())
        return shp

    return vm_impl


@vm_impl_getters.register(P.Squeeze)
def vm_impl_squeeze(self):
    """Generate vm_impl function for Squeeze"""

    def vm_impl(x):
        x = x.asnumpy()
        out = vm.squeeze(x, self.axis)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Transpose)
def vm_impl_transpose(self):
    """Generate vm_impl function for Transpose"""

    def vm_impl(x, perm=None):
        x = x.asnumpy()
        if perm is None:
            perm = [i for i in reversed(range(len(x.shape)))]
        out = vm.transpose(x, perm)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Split)
def vm_impl_split(self):
    """Generate vm_impl function for Split"""

    def vm_impl(x, axis, output_num):
        x = x.asnumpy()
        output = np.split(x, output_num, axis)
        output = [Tensor(value) for value in output]
        return output

    return vm_impl


@vm_impl_getters.register(P.Fill)
def vm_impl_fill(self):
    """Generate vm_impl function for Fill"""

    def vm_impl(dtype, dims, x):
        x_nptype = mstype.dtype_to_nptype(dtype)
        if isinstance(x, int):
            ret = np.full(dims, x, x_nptype)
        else:
            ret = np.full(dims, x, x_nptype)
        return Tensor(ret)

    return vm_impl


dtype_to_nptype_map = {
    30: np.bool_,
    32: np.int8,
    33: np.int16,
    34: np.int32,
    35: np.int64,
    37: np.uint8,
    38: np.uint16,
    39: np.uint32,
    40: np.uint64,
    42: np.float16,
    43: np.float32,
    44: np.float64
}


@vm_impl_getters.register(P.Eye)
def vm_impl_eye(self):
    """Generate vm_impl function for Eye"""

    def vm_impl(n, m, t):
        np_type = dtype_to_nptype_map[t]
        ret = np.eye(n, m, dtype=np_type)
        return Tensor(ret)

    return vm_impl


@vm_impl_getters.register(P.InvertPermutation)
def vm_impl_invert_permutation(self):
    """Generate vm_impl function for InvertPermutation"""

    def vm_impl(x):
        out = vm.invert_permutation(x)
        return out

    return vm_impl


@vm_impl_getters.register(P.Argmax)
def vm_impl_argmax(self):
    """Generate vm_impl function for Argmax"""

    def vm_impl(x):
        output = np.argmax(x.asnumpy(), axis=self.axis)
        return Tensor(output.ravel())

    return vm_impl


@vm_impl_getters.register(P.Tile)
def vm_impl_tile(self):
    """Generate vm_impl function for Tile"""

    def vm_impl(x, multiples):
        x = x.asnumpy()
        out = np.tile(x, multiples)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.ReduceAll)
def vm_impl_all(self):
    """Generate vm_impl function for All"""

    def vm_impl(x, axis, keep_dims):
        x = x.asnumpy()
        out = vm.all(x, axis, keep_dims)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.ReduceAny)
def vm_impl_any(self):
    """Generate vm_impl function for Any"""

    def vm_impl(x, axis, keep_dims):
        x = x.asnumpy()
        out = vm.any(x, axis, keep_dims)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Concat)
def vm_impl_concatV2(self):
    """Generate vm_impl function for Concat"""

    def vm_impl(x, axis):
        val_seq = [value.asnumpy() for value in x]
        out = np.concatenate(val_seq, axis)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Stack)
def vm_impl_stack(self):
    """Generate vm_impl function for Stack"""

    def vm_impl(x):
        val_seq = [value.asnumpy() for value in x]
        out = np.stack(val_seq, self.axis if not isinstance(self, str) else 0)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Unstack)
def vm_impl_unstack(self):
    """Generate vm_impl function for Unstack"""

    def vm_impl(x):
        x = x.asnumpy()
        out = np.moveaxis(x, self.axis if not isinstance(self, str) else 0, 0)
        output = [Tensor(value) for value in out]
        return output

    return vm_impl


@vm_impl_getters.register(P.Slice)
def vm_impl_slice(self):
    """Generate vm_impl function for Slice"""

    def vm_impl(x, begin, size):
        x = x.asnumpy()
        begin = begin.asnumpy()
        size = size.asnumpy()
        out = vm.Slice(x, begin, size)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(G.ConcatOffset)
def vm_impl_concatOffset(self):
    """Generate vm_impl function for ConcatOffset"""

    def vm_impl(x):
        out = vm.ConcatOffset(x)  # out is tuple
        return out

    return vm_impl


@vm_impl_getters.register(P.ReduceSum)
def vm_impl_sum(self):
    """Generate vm_impl function for Sum"""

    def vm_impl(x, axis, keep_dims, skip_mode):
        x = x.asnumpy()
        if axis == ():
            out = np.sum(x)
        else:
            out = np.sum(x, axis=axis)
        return Tensor(np.array(out))

    return vm_impl


@vm_impl_getters.register(P.Select)
def vm_impl_select(self):
    """Generate vm_impl function for Select"""

    def vm_impl(cond, x, y):
        """
        Args:
            cond: A `Tensor` of type `bool`
            x: A Tensor which may have the same shape as `condition`.
            y: A `Tensor` with the same shape and type as `x`.
        """
        cond = cond.asnumpy()
        x = x.asnumpy()
        y = y.asnumpy()
        out = vm.select(cond, x, y)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Square)
def vm_impl_square(self):
    """Generate vm_impl function for Square"""

    def vm_impl(x):
        x = x.asnumpy()
        return Tensor(x * x)

    return vm_impl


@vm_impl_getters.register(P.ZerosLike)
def vm_impl_zeros_like(self):
    """Generate vm_impl function for ZerosLike"""

    def vm_impl(x):
        return Tensor(np.zeros_like(x.asnumpy()))


@vm_impl_getters.register(P.Partial)
def vm_impl_partial(self):
    """Generate vm_impl function for Partial"""

    def vm_impl(*args):
        func = args[0].__call__
        partial_func = functools.partial(func, *args[1:])
        return partial_func

    return vm_impl


@vm_impl_getters.register(P.Depend)
def vm_impl_depend(self):
    """Generate vm_impl function for Depend"""

    def vm_impl(value, expr):
        return value

    return vm_impl


@vm_impl_getters.register(P.Load)
def vm_impl_load(self):
    """Generate vm_impl function for Load"""

    def vm_impl(value, u=None):
        return value

    return vm_impl


@vm_impl_getters.register(P.FillV2)
def vm_impl_fillv2(self):
    def vm_impl(x, y):
        if isinstance(x, Tensor):
            x = x.asnumpy()
        y = y.asnumpy()
        out = np.empty(x).astype(y.dtype)
        out.fill(y)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(SumExt)
def vm_impl_sum_ext(self):
    """Generate vm_impl function for SumExt"""

    def vm_impl(x, axis, keep_dims, dtype):
        x = x.asnumpy()
        nptype = None
        if dtype is not None:
            nptype = mstype.dtype_to_nptype(typing.type_id_to_type(dtype))
        if axis is None or axis == ():
            out = np.sum(x, keepdims=keep_dims, dtype=nptype)
        else:
            out = np.sum(x, axis=axis, keepdims=keep_dims, dtype=nptype)
        return Tensor(np.array(out))

    return vm_impl

