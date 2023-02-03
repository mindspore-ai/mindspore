# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""The names of functional part are summarized here."""

import numpy as np

import mindspore as ms
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common import Tensor
from mindspore.common._decorator import deprecated
from mindspore.common import dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.ops import _constants
from mindspore.ops.function import *
from mindspore.ops.function.sparse_func import sparse_add
from mindspore.ops.primitive import constexpr, Primitive
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops
from mindspore.ops.operations import _csr_ops
from mindspore.ops.operations import _inner_ops
from mindspore.ops.operations import linalg_ops
from mindspore.ops.operations.math_ops import Median
from mindspore.ops.operations.array_ops import UniqueConsecutive
from mindspore.ops.operations.nn_ops import AdaptiveMaxPool2D
from mindspore.ops.composite import _Vmap, Shard

typeof = Primitive('typeof')
hastype = Primitive('hastype')
cast = P.Cast()
dtype = P.DType()
isconstant = Primitive('is_constant')
isconstant.set_const_prim(True)

issubclass_ = P.IsSubClass()
isinstance_ = P.IsInstance()

merge = P.Merge()
geswitch = P.GeSwitch()
strided_slice = P.StridedSlice()
check_bprop = P.CheckBprop()
reduce_sum = P.ReduceSum()
reduce_max = P.ReduceMax()
reduce_min = P.ReduceMin()
reduce_mean = P.ReduceMean()
sort = P.Sort()
tensor_range = P.Range()
tensor_scatter_update = P.TensorScatterUpdate()
scatter_nd_update = P.ScatterNdUpdate()
mixed_precision_cast = _inner_ops.MixedPrecisionCast()


def pack(x):
    """Call stack in this pack function."""
    print("WARNING: 'pack' is deprecated from version 1.1 and will be removed in a future version, use 'stack' instead"
          ".")
    return stack(x)


partial = P.Partial()
# depend: mount a node to another node
depend = P.Depend()
identity = P.identity()
shard_fn = Shard()


def shard(fn, in_strategy, out_strategy, parameter_plan=None, device="Ascend", level=0):
    """Apply distributed process for fn"""
    if not isinstance(fn, ms.nn.Cell):
        raise TypeError(f"Type of fn must be 'Cell', but got type {type(fn)}")
    return shard_fn(fn, in_strategy, out_strategy, parameter_plan, device, level)


@deprecated("1.8", "range", False)
def arange(start=0, stop=None, step=1, rtype=None):
    r"""
    The ops.arange interface is deprecated, please use :func:`mindspore.ops.range`

    Supported Platforms:
        deprecated
    """
    if stop is None:
        start, stop = 0, start

    arg_map = {"start": start, "stop": stop, "step": step}
    for arg in ("start", "stop", "step"):
        arg_value = arg_map.get(arg)
        if not isinstance(arg_value, int) and not isinstance(arg_value, float):
            _raise_arange_type_error(arg, arg_value)
    if start >= stop:
        _raise_arange_value_error(start, stop)

    if rtype is None:
        if isinstance(start, float) or isinstance(stop, float) or isinstance(step, float):
            rtype = mstype.float32
        else:
            rtype = mstype.int32
    data = _arange(start, stop, step)
    return _make_tensor(data, rtype)


@constexpr
def _make_tensor(data, rtype):
    """Make Tensor"""
    return Tensor(data, dtype=rtype)


@constexpr
def _arange(start, stop, step):
    """Arange compute"""
    return np.arange(start, stop, step)


@constexpr
def _raise_arange_type_error(arg, arg_value):
    """
    Raise TypeError in both graph/pynative mode.
    """
    raise TypeError("For mindspore.ops.arange, the argument '{}' must be int or float, but got {}."
                    .format(arg, type(arg_value)))


@constexpr
def _raise_arange_value_error(start, stop):
    """
    Raise TypeError in both graph/pynative mode
    """
    raise ValueError("For mindspore.ops.arange, the argument 'start' must be < 'stop', but got 'start': {}, "
                     "'stop': {}.".format(start, stop))


def narrow(inputs, axis, start, length):
    """
    Returns a narrowed tensor from input tensor.
    The dimension axis is input from start to start + length.

    Args:
        inputs (Tensor): the tensor to narrow.
        axis (int): the axis along which to narrow.
        start (int): the starting dimension.
        length (int): the distance to the ending dimension.

    Returns:
        Tensor.

        - output (Tensors) - The narrowed tensor.

    Raises:
        TypeError: If the input is not a tensor or tuple or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.int32)
        >>> output = ops.narrow(x, 0, 0, 2)
        >>> print(output)
        [[ 1 2 3]
         [ 4 5 6]]
        >>> output = ops.narrow(x, 1, 1, 2)
        >>> print(output)
        [[ 2 3]
         [ 5 6]
         [ 8 9]]
    """
    validator.check_axis_in_range(axis, inputs.ndim)
    validator.check_int_range(start, 0, inputs.shape[axis], Rel.INC_LEFT)
    validator.check_int_range(length, 1, inputs.shape[axis] - start, Rel.INC_BOTH)

    begins = [0] * inputs.ndim
    begins[axis] = start
    sizes = [i for i in inputs.shape]
    sizes[axis] = length
    return P.Slice()(inputs, begins, sizes)


vmap_instance = _Vmap()


def vmap(fn, in_axes=0, out_axes=0):
    r"""
    Vectorizing map (vmap) is a kind of higher-order function to map `fn` along the parameter axes.

    Vmap is pioneered by Jax and it removes the restriction of batch dimension on the operator, and provides a
    more convenient and unified operator expression. Moreover, it allows users to composite with other functional
    modules such as :func:`mindspore.ops.grad`, to improve the development efficiency. In addition, the vectorizing
    map does not execute loops outside the function, but sinks loops into the primitive operations of the function
    for better performance. When combined with `Graph Kernel Fusion`, operational efficiency would be further improved.

    .. warning::
        This is an experimental prototype that is subject to change and/or delete.

    Note:
        1. The power of vmap comes from the implementation of VmapRules of primitives. Although we have designed a
        generalized rule for user custom operators, we can not guarantee that it works well for all operators,
        please be aware the risk of use. If you want to achieve a better performance, please refer to the tutorial to
        implement the specific VmapRule for the custom operator, which won't take too much time.
        2. When calling the random number generation methods within the scope of vmap, the same random number is
        generated among vector functions each time. If you expect each vector branch to use different random numbers,
        you need to generate batch random numbers externally in advance and then transfer them to vmap.

    Args:
        fn (Union[Cell, Function]): Function to be mapped along the parameter axes, which takes at least one argument
            and returns one or more Tensors or the type of data supported by the MindSpore Tensor.
        in_axes (Union[int, list, tuple]): Specifies which dimensions (axes) of the inputs should be mapped over.
            If `in_axes` is an integer, all arguments of `fn` are mapped over according to this axis index. If `in_axes`
            is a tuple or list, which only composed of integers or Nones and the length should equal to the number of
            positional arguments to `fn`, indicates which axis to map for each corresponding positional argument.
            Note that, axis integers must be in range :math:`[-ndim, ndim)` for each argument, where `ndim` is the
            number of dimensions of the corresponding argument.  None means not mapping along any axis. Also the
            mapping axis index of the `in_axes` must have at least one positional parameter not None. The sizes of
            the mapped axes (`axis_size`) for all arguments must be equal. Default: 0.
        out_axes (Union[int, list, tuple]): Specifies where the mapped dimensions (axes) should appear in the
            outputs. If `out_axes` is an integer, all outputs of `fn` are specified according to this axis. If
            `out_axes` is a tuple or list, which only composed of integers or Nones. And its length also should be equal
            to the number of outputs of `fn`. Note that, axis integers must be in range :math:`[-ndim, ndim)` for each
            output, where `ndim` is the dimension of the output of the `vmap`-mapped function. All outputs with a
            non-None mapped axis must specify a non-None `out_axes`, and if outputs with None mapped axis specifies
            a non-None `out_axes`, the result broadcasts across the mapped axis. Default: 0.

    Returns:
        Function, returns the Vectorized/Batched version function of `fn`. The arguments and outputs of this function
        correspond to those of `fn`, but it adds an extra batch dimension at positions specified by `in_axes` and
        `out_axes`.

    Raises:
        RuntimeError: If base elements in `in_axes` or `out_axes` are not a None or an integer.
            If the all base elements in `in_axes` or `out_axes` are None.
            If `in_axes` is not single integer, and the length of `in_axes` is not equal to the arguments sizes.
            If `out_axes` is not single integer, and the length of `out_axes` is not equal to the outputs sizes.
            If the `axis_size` of each arguments in the scope of `vmap` are not equal.
            If the axis in `in_axes` or `out_axes` is out of bounds.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import vmap
        >>> def test_vmap(x, y, z):                                              # ([a],[a],[a]) -> [a]
        ...     return x + y + z
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32))    # [b, a]
        >>> y = Tensor(np.array([[-3, -2, -1], [3, 2, 1]]).astype(np.float32))   # [a, b]
        >>> z = Tensor(np.array([0, 3]).astype(np.float32))                      # [a]
        >>> output = vmap(test_vmap, in_axes=(0, 1, None), out_axes=1)(x, y, z)  # ([b, a],[a, b],[a]) -> [a, b]
        >>> print(output)
        [[-2  1  4]
         [ 8  9 10]]
    """
    return vmap_instance(fn, in_axes, out_axes)


tuple_setitem = Primitive('tuple_setitem')
tuple_getitem = Primitive(_constants.kTupleGetItem)
list_getitem = Primitive('list_getitem')
list_setitem = Primitive('list_setitem')
dict_getitem = Primitive('dict_getitem')
dict_setitem = Primitive('dict_setitem')
tuple_div = Primitive("tuple_div")
tuple_len = Primitive("tuple_len")
list_len = Primitive("list_len")
tuple_reversed = Primitive("tuple_reversed")
make_range = Primitive("make_range")
make_tuple = Primitive('MakeTuple')
make_dict = Primitive('make_dict')
make_list = Primitive('make_list')
make_slice = Primitive('make_slice')
tuple_equal = Primitive("tuple_equal")
list_equal = Primitive("list_equal")

scalar_add = Primitive(_constants.kScalarAdd)
scalar_mul = Primitive(_constants.kScalarMul)
scalar_sub = Primitive(_constants.kScalarSub)
scalar_div = Primitive(_constants.kScalarDiv)
scalar_floordiv = Primitive(_constants.kScalarFloordiv)
scalar_log = Primitive('scalar_log')
scalar_pow = Primitive(_constants.kScalarPow)
scalar_gt = Primitive('scalar_gt')
scalar_ge = Primitive('scalar_ge')
scalar_le = Primitive('scalar_le')
scalar_lt = Primitive('scalar_lt')
scalar_eq = Primitive('scalar_eq')
scalar_ne = Primitive('scalar_ne')
scalar_uadd = Primitive(_constants.kScalarUadd)
scalar_usub = Primitive(_constants.kScalarUsub)
scalar_mod = Primitive(_constants.kScalarMod)
string_eq = Primitive('string_eq')
string_concat = Primitive('string_concat')
bool_not = Primitive("bool_not")
bool_or = Primitive("bool_or")
bool_and = Primitive("bool_and")
bool_eq = Primitive("bool_eq")
cumsum = P.CumSum()
cumprod = P.CumProd()
array_to_scalar = Primitive('array_to_scalar')
is_ = Primitive("is_")
is_not = Primitive("is_not")
in_dict = Primitive("in_dict")
not_in_dict = Primitive("not_in_dict")
broadcast_gradient_args = Primitive('BroadcastGradientArgs')
array_reduce = Primitive('array_reduce')
zeros = P.Zeros()
zeros_like = P.ZerosLike()
distribute = Primitive('distribute')
embed = Primitive('embed')
ref_to_embed = _grad_ops.RefToEmbed()
environ_create = Primitive('EnvironCreate')
environ_set = Primitive('EnvironSet')
environ_get = Primitive('EnrironGet')
environ_add = Primitive('EnvironAdd')
J = Primitive('J')
SliceGetItem = Primitive("SliceGetItem")
switch = Primitive('Switch')
switch_layer = Primitive('switch_layer')
# for sum bprop
reduced_shape = Primitive("reduced_shape")
# shape_mul:input must be shape multiply elements in tuple(shape)
shape_mul = Primitive("shape_mul")
# a primitive to compare between tuple.
stop_gradient = Primitive("stop_gradient")

tensor_operator_registry.register('add', P.Add)
tensor_operator_registry.register('addr', addr)
tensor_operator_registry.register('addcdiv', P.Addcdiv)
tensor_operator_registry.register('addcmul', P.Addcmul)
tensor_operator_registry.register('all', P.ReduceAll)
tensor_operator_registry.register('any', P.ReduceAny)
tensor_operator_registry.register('atan2', atan2)
tensor_operator_registry.register('abs', P.Abs)
tensor_operator_registry.register('sqrt', sqrt)
tensor_operator_registry.register('square', square)
tensor_operator_registry.register('sub', sub)
tensor_operator_registry.register('tan', P.Tan)
tensor_operator_registry.register('acos', acos)
tensor_operator_registry.register('cos', cos)
tensor_operator_registry.register('acosh', acosh)
tensor_operator_registry.register('cosh', P.Cosh)
tensor_operator_registry.register('asin', asin)
tensor_operator_registry.register('pow', P.Pow)
tensor_operator_registry.register('amin', amin)
tensor_operator_registry.register('amax', amax)
tensor_operator_registry.register('mean', P.ReduceMean)
tensor_operator_registry.register('prod', prod)
tensor_operator_registry.register('round', P.Round)
tensor_operator_registry.register('reshape', P.Reshape)
tensor_operator_registry.register('reverse_sequence', P.ReverseSequence)
tensor_operator_registry.register('xlogy', P.Xlogy)
tensor_operator_registry.register('flatten', P.Flatten)
tensor_operator_registry.register('transpose', P.Transpose)
tensor_operator_registry.register('broadcast_to', P.BroadcastTo)
tensor_operator_registry.register('matmul', matmul)
tensor_operator_registry.register('xdivy', P.Xdivy)
tensor_operator_registry.register('argmax', P.Argmax)
tensor_operator_registry.register('cumsum', P.CumSum)
tensor_operator_registry.register('cummin', cummin)
tensor_operator_registry.register('cummax', cummax)
tensor_operator_registry.register('index_fill', index_fill)
tensor_operator_registry.register('bitwise_and', bitwise_and)
tensor_operator_registry.register('bitwise_or', bitwise_or)
tensor_operator_registry.register('bitwise_xor', bitwise_xor)
tensor_operator_registry.register('ger', ger)
tensor_operator_registry.register('reduce_max', P.ReduceMax)
tensor_operator_registry.register('reduce_min', P.ReduceMin)
tensor_operator_registry.register('random_categorical', random_categorical)
tensor_operator_registry.register('maximum', P.Maximum)
tensor_operator_registry.register('mirror_pad', P.MirrorPad)
tensor_operator_registry.register('minimum', P.Minimum)
tensor_operator_registry.register('matrix_determinant', matrix_determinant)
tensor_operator_registry.register('log1p', log1p)
tensor_operator_registry.register('log_matrix_determinant', log_matrix_determinant)
tensor_operator_registry.register('ceil', P.Ceil)
tensor_operator_registry.register('fill', P.Fill)
tensor_operator_registry.register('tile', P.Tile)
tensor_operator_registry.register('logical_not', P.LogicalNot)
tensor_operator_registry.register('logit', logit)
tensor_operator_registry.register('sum', P.ReduceSum)
tensor_operator_registry.register('split', P.Split)
tensor_operator_registry.register('select', P.Select)
tensor_operator_registry.register('zeros_like', P.ZerosLike)
tensor_operator_registry.register('scalar_to_tensor', scalar_to_tensor)
tensor_operator_registry.register('stop_gradient', stop_gradient)
tensor_operator_registry.register('masked_fill', masked_fill)
tensor_operator_registry.register('masked_select', masked_select)
tensor_operator_registry.register('nonzero', nonzero)
tensor_operator_registry.register('isclose', isclose)
tensor_operator_registry.register('inv', inv)
tensor_operator_registry.register('invert', invert)
tensor_operator_registry.register('hardshrink', P.HShrink)
tensor_operator_registry.register('soft_shrink', P.SoftShrink)
tensor_operator_registry.register('svd', linalg_ops.Svd)
tensor_operator_registry.register('diag', P.Diag)
tensor_operator_registry.register('unique_consecutive', UniqueConsecutive)
tensor_operator_registry.register('unique_with_pad', P.UniqueWithPad)
tensor_operator_registry.register('inplace_update', P.InplaceUpdate)
tensor_operator_registry.register('col2im', col2im)
tensor_operator_registry.register('standard_laplace', P.StandardLaplace)
tensor_operator_registry.register('split', P.Split)
tensor_operator_registry.register('erf', P.Erf)
tensor_operator_registry.register('erfc', P.Erfc)
tensor_operator_registry.register('standard_normal', P.StandardNormal)
tensor_operator_registry.register('sigmoid', P.Sigmoid)
tensor_operator_registry.register('median', Median)
tensor_operator_registry.register('tanh', tanh)
tensor_operator_registry.register('exp', P.Exp)
tensor_operator_registry.register('addmv', addmv)
tensor_operator_registry.register('asinh', asinh)
tensor_operator_registry.register('atan', atan)
tensor_operator_registry.register('atanh', atanh)
tensor_operator_registry.register('bmm', bmm)
# ms cannot support Tensor(True) compare
tensor_operator_registry.register('__eq__', equal)
tensor_operator_registry.register('__ne__', not_equal)
tensor_operator_registry.register('__neg__', neg_tensor)
tensor_operator_registry.register('__lt__', tensor_lt)
tensor_operator_registry.register('__le__', tensor_le)
tensor_operator_registry.register('__gt__', tensor_gt)
tensor_operator_registry.register('__ge__', tensor_ge)
tensor_operator_registry.register('__logical_not__', logical_not)
tensor_operator_registry.register('shape', shape)
tensor_operator_registry.register('squeeze', squeeze)
tensor_operator_registry.register('expand_dims', expand_dims)
# support GE backend for no compare operators
tensor_operator_registry.register('cast', cast)
tensor_operator_registry.register('shape_mul', shape_mul)
tensor_operator_registry.register('fill', fill)
tensor_operator_registry.register('fills', fills)
tensor_operator_registry.register('concatenate', P.Concat)
tensor_operator_registry.register('eye', eye)
tensor_operator_registry.register('reduce_sum', reduce_sum)
tensor_operator_registry.register('tensor_slice', tensor_slice)
tensor_operator_registry.register('select', select)
tensor_operator_registry.register('gather', gather)
tensor_operator_registry.register('gather_d', gather_d)
tensor_operator_registry.register('gather_elements', gather_elements)
tensor_operator_registry.register('gather_nd', gather_nd)
tensor_operator_registry.register('stack', stack)
tensor_operator_registry.register('unstack', unstack)
tensor_operator_registry.register('log', log)
tensor_operator_registry.register('lerp', lerp)
tensor_operator_registry.register('floor', floor)
# support sparse tensor operators
tensor_operator_registry.register('csr_add', csr_add)
tensor_operator_registry.register('csr_mul', csr_mul)
tensor_operator_registry.register('csr2coo', csr2coo)
tensor_operator_registry.register('coo2csr', coo2csr)
tensor_operator_registry.register('csr_div', csr_div)
tensor_operator_registry.register('csr_mv', csr_mv)
tensor_operator_registry.register('csr_mm', _csr_ops.CSRMM)
tensor_operator_registry.register('csr_reduce_sum', csr_reduce_sum)
tensor_operator_registry.register('dense_to_sparse_csr', dense_to_sparse_csr)
tensor_operator_registry.register('dense_to_sparse_coo', dense_to_sparse_coo)
tensor_operator_registry.register('csr_to_dense', csr_to_dense)
tensor_operator_registry.register('narrow', narrow)
tensor_operator_registry.register('sort', sort)
tensor_operator_registry.register('csr_to_coo', csr_to_coo)
tensor_operator_registry.register('zeros', zeros)
tensor_operator_registry.register('unsorted_segment_min', unsorted_segment_min)
tensor_operator_registry.register('unsorted_segment_max', unsorted_segment_max)
tensor_operator_registry.register('unsorted_segment_prod', unsorted_segment_prod)
tensor_operator_registry.register('tensor_scatter_update', tensor_scatter_update)
tensor_operator_registry.register('tensor_scatter_mul', tensor_scatter_mul)
tensor_operator_registry.register('tensor_scatter_div', tensor_scatter_div)
tensor_operator_registry.register('tensor_scatter_min', P.TensorScatterMin)
tensor_operator_registry.register('tensor_scatter_max', P.TensorScatterMax)
tensor_operator_registry.register('tensor_scatter_sub', tensor_scatter_sub)
tensor_operator_registry.register('tensor_scatter_add', tensor_scatter_add)
tensor_operator_registry.register('bernoulli', bernoulli)
tensor_operator_registry.register('norm', norm)
tensor_operator_registry.register('renorm', renorm)
tensor_operator_registry.register('adaptive_max_pool2d', AdaptiveMaxPool2D)
tensor_operator_registry.register('coalesce', coalesce)
tensor_operator_registry.register('argmax_with_value', max)
tensor_operator_registry.register('argmin_with_value', min)
tensor_operator_registry.register('coo_add', sparse_add)
tensor_operator_registry.register('top_k', P.TopK)
tensor_operator_registry.register('isfinite', P.IsFinite)
__all__ = [name for name in dir() if name[0] != "_"]
__all__.remove('Primitive')
