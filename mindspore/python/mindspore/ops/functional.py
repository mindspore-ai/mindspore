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

from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common import ms_function
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.nn.grad.cell_grad import _JvpInner
from mindspore.nn.grad.cell_grad import _VjpInner
from mindspore.ops import _constants
from mindspore.ops.function import *
from mindspore.ops.primitive import constexpr, Primitive
from . import operations as P
from .operations import _grad_ops
from .operations import _csr_ops
from .operations import linalg_ops
from .operations.array_ops import UniqueConsecutive
from .operations.nn_ops import AdaptiveMaxPool2D
from .function.sparse_func import sparse_add
from .composite import _Grad, Shard, _Vmap, _TaylorOperation
from .._c_expression import security

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
square = P.Square()
sqrt = P.Sqrt()
reduce_sum = P.ReduceSum()
reduce_max = P.ReduceMax()
reduce_min = P.ReduceMin()
reduce_mean = P.ReduceMean()
sort = P.Sort()
tensor_range = P.Range()
if not security.enable_security():
    print_ = P.Print()
tensor_scatter_update = P.TensorScatterUpdate()
scatter_nd_update = P.ScatterNdUpdate()


def pack(x):
    """Call stack in this pack function."""
    print("WARNING: 'pack' is deprecated from version 1.1 and will be removed in a future version, use 'stack' instead"
          ".")
    return stack(x)


partial = P.Partial()
# depend: mount a node to another node
depend = P.Depend()
identity = P.identity()


@constexpr
def _convert_grad_position_type(grad_position):
    """Check and convert the type and size of grad position index."""
    if isinstance(grad_position, tuple):
        for gp in grad_position:
            if not isinstance(gp, int):
                raise TypeError(f"For 'F.grad', the element in 'grad_position' must be int.")
            if gp < 0:
                raise ValueError("The element in grad_position must be >= 0.")
    elif isinstance(grad_position, int):
        if grad_position < 0:
            raise ValueError("grad_position must be >= 0.")
        grad_position = (grad_position,)
    else:
        raise TypeError(f"For 'F.grad', the 'grad_position' must be int or tuple.")
    return grad_position


grad_by_position = _Grad(get_by_list=False, sens_param=False, get_by_position=True)
grad_by_position_with_sens = _Grad(get_by_list=False, sens_param=True, get_by_position=True)


def grad(fn, grad_position=0, sens_param=False):
    r"""
    A wrapper function to generate the gradient function for the input function.

    Args:
        fn (Union(Cell, function)): Function to do GradOperation.
        grad_position (Union(int, tuple[int])): If int, get the gradient with respect to single input.
            If tuple, get the gradients with respect to selected inputs. 'grad_position' begins with 0. Default: 0.
        sens_param (bool): Whether to append sensitivity (gradient with respect to output) as input.
            If sens_param is False, a 'ones_like(outputs)' sensitivity will be attached automatically. Default: False.

    Returns:
        Function, returns the gradient function for the input function or cell.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.ops.functional import grad
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x*y*z
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
        >>> z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
        >>> net = Net()
        >>> output = grad(net, grad_position=(1, 2))(x, y, z)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 0.00000000e+00,  6.00000000e+00],
         [ 1.50000000e+01, -4.00000000e+00]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[-2.00000000e+00,  6.00000000e+00],
         [-3.00000000e+00,  8.00000000e+00]]))
    """
    grad_position = _convert_grad_position_type(grad_position)
    if sens_param:
        return grad_by_position_with_sens(fn, None, grad_position)
    return grad_by_position(fn, None, grad_position)


def _trans_jet_inputs(primals_item, series_item):
    """Trans inputs of jet"""
    value_type = [mstype.int32, mstype.int64, mstype.float32, mstype.float64]
    if not dtype(primals_item) in value_type or dtype(primals_item) != dtype(series_item):
        raise TypeError(f"For `F.jet`, the elements' types of primals and series must be the same and belong to "
                        f"`mstype.int32, mstype.int64, mstype.float32, mstype.float64`, but got other dtype.")
    if dtype(primals_item) in [mstype.int32, mstype.int64]:
        return cast(primals_item, mstype.float32), cast(series_item, mstype.float32)
    return primals_item, series_item


def _check_jet_inputs(primals, series):
    """Check inputs of jet"""
    if not (isinstance(primals, Tensor) and isinstance(series, Tensor)) and \
            not (isinstance(primals, tuple) and isinstance(series, tuple)):
        raise TypeError(f"For 'F.jet', the 'primals' and `series` must be both Tensor or tuple.")
    if isinstance(primals, Tensor):
        if primals.shape == series.shape[1:]:
            return _trans_jet_inputs(primals, series)
        if primals.shape == series.shape:
            return _trans_jet_inputs(primals, series.expand_dims(axis=0))
        raise ValueError("In series, the shape of each element must be the same as the primals.")
    if len(primals) != len(series):
        raise ValueError("The lengths of primals and series must be the same.")
    check_primals = []
    check_series = []
    for i, j in zip(primals, series):
        trans_primals_item, trans_series_item = _trans_jet_inputs(i, j)
        check_primals.append(trans_primals_item)
        check_series.append(trans_series_item)
    return check_primals, check_series


_taylor = _TaylorOperation()


def _preprocess_jet(x, y):
    concat_op = P.Concat()
    return concat_op((expand_dims(x, 0), y))


def jet(fn, primals, series):
    """
    This function is designed to calculate the higher order differentiation of given composite function. To figure out
    first to `n`-th order differentiations, original inputs and first to `n`-th order derivative of original inputs
    must be provided together. Generally, it is recommended to set the values of given first order derivative to 1,
    while the other to 0, which is like the derivative of origin input with respect to itself.

    Args:
        fn (Union[Cell, function]): Function to do TaylorOperation.
        primals (Union[Tensor, tuple[Tensor]]): The inputs to `fn`.
        series (Union[Tensor, tuple[Tensor]]): If tuple, the length and type of series should be the same as inputs.
            For each Tensor, the length of first dimension `i` represents the `1` to `i+1`-th order of derivative of
            output with respect to the inputs will be figured out.

    Returns:
        Tuple, tuple of out_primals and out_series.

        - **out_primals** (Union[Tensor, list[Tensor]]) - The output of `fn(primals)`.
        - **out_series** (Union[Tensor, list[Tensor]]) - The `1` to `i+1`-th order of derivative of output with respect
          to the inputs.

    Raises:
        TypeError: If `primals` is not a tensor or tuple of tensors.
        TypeError: If type of `primals` is not the same as type of `series`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> import mindspore as ms
        >>> import mindspore.ops as P
        >>> from mindspore import Tensor
        >>> from mindspore.ops.functional import jet
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.sin = P.Sin()
        ...         self.exp = P.Exp()
        ...     def construct(self, x):
        ...         out1 = self.sin(x)
        ...         out2 = self.exp(out1)
        ...         return out2
        >>> primals = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> series = Tensor(np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]).astype(np.float32))
        >>> net = Net()
        >>> out_primals, out_series = jet(net, primals, series)
        >>> print(out_primals, out_series)
        [[2.319777  2.4825778]
         [1.1515628 0.4691642]] [[[ 1.2533808  -1.0331168 ]
          [-1.1400385  -0.3066662 ]]
         [[-1.2748207  -1.8274734 ]
          [ 0.966121    0.55551505]]
         [[-4.0515366   3.6724353 ]
          [ 0.5053504  -0.52061415]]]
    """
    primals, series = _check_jet_inputs(primals, series)
    derivative_fn = _taylor(fn)
    if isinstance(primals, list) and list_len(primals) > 1:
        inputs = map(_preprocess_jet, primals, series)
        outputs = derivative_fn(*inputs)
    else:
        inputs = _preprocess_jet(primals, series)
        outputs = derivative_fn(inputs)
    if isinstance(outputs, tuple) and tuple_len(outputs) > 1:
        out_primals = []
        out_series = []
        for element in outputs:
            out_primals.append(element[0])
            out_series.append(element[1:])
    else:
        out_primals = outputs[0]
        out_series = outputs[1:]
    return out_primals, out_series


def _trans_derivative_inputs(primals_item):
    """Trans inputs of derivative"""
    value_type = [mstype.int32, mstype.int64, mstype.float32, mstype.float64]
    if not dtype(primals_item) in value_type:
        raise TypeError(f"For `F.derivative`, the elements of primals must belong to "
                        f"`mstype.int32, mstype.int64, mstype.float32, mstype.float64`, but got other dtype.")
    if dtype(primals_item) in [mstype.int32, mstype.int64]:
        return cast(primals_item, mstype.float32)
    return primals_item


@constexpr
def _check_derivative_order(order):
    """check input order of derivative"""
    if not isinstance(order, int):
        raise TypeError(f"For `F.derivative`, the type of order must be int.")
    if order < 1:
        raise ValueError(f"For `F.derivative`, value of order should not be less than 1, but got {order}.")
    return True


def _preprocess_derivate_order_one(x):
    concat_op = P.Concat()
    return concat_op((expand_dims(x, 0), ones((1,) + x.shape, dtype(x))))


def _preprocess_derivate_order_more(x, order):
    concat_op = P.Concat()
    return concat_op((x, zeros((order - 1,) + x[0].shape, dtype(x))))


def derivative(fn, primals, order):
    """
    This function is designed to calculate the higher order differentiation of given composite function. To figure out
    `order`-th order differentiations, original inputs and order must be provided together. In particular, the value of
    input first order derivative is set to 1, while the other to 0.

    Args:
        fn (Union[Cell, function]): Function to do TaylorOperation.
        primals (Union[Tensor, tuple[Tensor]]): The inputs to `fn`.
        order (int): For each Tensor, the `order`-th order of derivative of output with respect to the inputs will be
            figured out.

    Returns:
        Tuple, tuple of out_primals and out_series.

        - **out_primals** (Union[Tensor, list[Tensor]]) - The output of `fn(primals)`.
        - **out_series** (Union[Tensor, list[Tensor]]) - The `order`-th order of derivative of output with respect
          to the inputs.

    Raises:
        TypeError: If `primals` is not a tensor or tuple of tensors.
        TypeError: If `order` is not int.
        ValueError: If `order` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as P
        >>> from mindspore import Tensor
        >>> from mindspore.ops.functional import derivative
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.sin = P.Sin()
        ...         self.exp = P.Exp()
        ...     def construct(self, x):
        ...         out1 = self.sin(x)
        ...         out2 = self.exp(out1)
        ...         return out2
        >>> primals = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> order = 3
        >>> net = Net()
        >>> out_primals, out_series = derivative(net, primals, order)
        >>> print(out_primals, out_series)
        [[2.319777  2.4825778]
         [1.1515628 0.4691642]] [[-4.0515366   3.6724353 ]
         [ 0.5053504  -0.52061415]]
    """
    derivative_fn = _taylor(fn)
    concat_op = P.Concat()
    series_one = 1
    _check_derivative_order(order)
    if isinstance(primals, tuple):
        trans_primals = map(_trans_derivative_inputs, primals)
        inputs = map(_preprocess_derivate_order_one, trans_primals)
        if order > 1:
            processed_inputs = []
            for element in inputs:
                processed_inputs.append(_preprocess_derivate_order_more(element, order))
            outputs = derivative_fn(*processed_inputs)
        else:
            outputs = derivative_fn(*inputs)
    else:
        primals = _trans_derivative_inputs(primals)
        series = zeros((order,) + primals.shape, dtype(primals))
        series[0] = series_one
        inputs = concat_op((expand_dims(primals, 0), series))
        outputs = derivative_fn(inputs)
    if isinstance(outputs, tuple) and tuple_len(outputs) > 1:
        out_primals = []
        out_series = []
        for element in outputs:
            out_primals.append(element[0])
            out_series.append(element[-1])
    else:
        out_primals = outputs[0]
        out_series = outputs[-1]
    return out_primals, out_series


def jvp(fn, inputs, v):
    """
    Compute the jacobian-vector-product of the given network.

    Args:
        fn (Union[Function, Cell]): The function or net that takes Tensor inputs and returns single tensor or tuple of
            Tensors.
        inputs (Union[Tensor, Tuple or List of Tensors]): The inputs to `fn`.
        v (Union[Tensor, Tuple or or List of Tensors]): The shape and type of v should be the same as inputs.

    Returns:
        Tuple, tuple of output and jvp.

        - **netout** (Tensor or Tuple of Tensors) - The output of "fn(inputs)".
        - **jvp** (Tensor or Tuple of Tensors) - The result of the dot product.

    Raises:
        TypeError: If the input is not a tensor or tuple or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = F.jvp(Net(), (x, y), (v, v))
        >>> print(output[0])
        [[ 2. 10.]
         [30. 68.]]
        >>> print(output[1])
        [[ 4. 13.]
         [28. 49.]]
    """
    jvp_inner = _JvpInner()

    @ms_function(hash_args=fn)
    def _wrap_container(*arg):
        args = arg[1:]
        vectors = arg[0]
        return jvp_inner(fn, vectors, *args)

    if not isinstance(inputs, (Tensor, tuple, list)) or not isinstance(v, (Tensor, tuple, list)):
        _raise_type_error()
    if isinstance(v, list):
        v = tuple(v)
    if isinstance(inputs, (tuple, list)):
        return _wrap_container(v, *inputs)
    return _wrap_container(v, inputs)


def vjp(fn, inputs, v):
    """
    Compute the vector-jacobian-product of the given network.

    Args:
        fn (Union[Function, Cell]): The function or net that takes Tensor inputs and returns single tensor or tuple of
            Tensors.
        inputs (Union[Tensor, Tuple or List of Tensors]): The inputs to `fn`.
        v (Union[Tensor, Tuple or List of Tensors]): The shape and type of v should be the same as outputs.

    Returns:
        Tuple, tuple of output and vjp.

        - **netout** (Tensor or Tuple of Tensors) - The output of "fn(inputs)".
        - **vjp** (Tensor or Tuple of Tensors) - The result of the dot product.

    Raises:
        TypeError: If the input is not a tensor or tuple or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = F.vjp(Net(), (x, y), v)
        >>> print(output[0])
        [[ 2. 10.]
         [30. 68.]]
        >>> print(output[1])
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 3.00000000e+00,  1.20000000e+01],
         [ 2.70000000e+01,  4.80000000e+01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.00000000e+00,  1.00000000e+00],
         [ 1.00000000e+00,  1.00000000e+00]]))
    """
    vjp_inner = _VjpInner()

    @ms_function(hash_args=fn)
    def wrap_container(*arg):
        args = arg[:-1]
        vectors = arg[-1]
        return vjp_inner(fn, *args, vectors)

    if not isinstance(inputs, (Tensor, tuple, list)) or not isinstance(v, (Tensor, tuple, list)):
        _raise_type_error()
    if isinstance(v, list):
        v = tuple(v)
    if isinstance(inputs, (tuple, list)):
        return wrap_container(*inputs, v)
    return wrap_container(inputs, v)


shard_fn = Shard()


def shard(fn, in_strategy, out_strategy, device="Ascend", level=0):
    return shard_fn(fn, in_strategy, out_strategy, device, level)


def arange(start=0, stop=None, step=1, rtype=None):
    """
    Returns evenly spaced values within a given interval.

    Args:
        start(Union[int, float]): Start value of interval. The interval includes this value. When
            `stop` is None, `start` must be greater than 0, and the interval is :math:`[0, start)`.
            When `stop` is not None, `start` must be less than `stop`.
        stop(Union[int, float], optional): End value of interval. The interval does not
            include this value. Default is None.
        step(Union[int, float], optional): Spacing between values. For any output
            `out`, this is the distance between two adjacent values, :math:`out[i+1] - out[i]`.
            The default step size is 1. If `step` is specified as a position argument,
            `start` must also be given.
        rtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor type.
            If rtype is None, the data type of the new tensor will be inferred from start,
            stop and step. Default is None.

    Returns:
        Tensor with evenly spaced values.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input arguments have values not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.ops as ops
        >>> print(ops.arange(0, 5, 1))
        [0 1 2 3 4]
        >>> print(ops.arange(3))
        [0 1 2]
        >>> print(ops.arange(start=0, stop=3))
        [0 1 2]
        >>> print(ops.arange(0, stop=3, step=0.5))
        [0.  0.5 1.  1.5 2.  2.5]
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
        >>> from mindspore.ops import functional as F
        >>> from mindspore import Tensor
        >>> x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.int32)
        >>> output = F.narrow(x, 0, 0, 2)
        >>> print(output)
        [[ 1 2 3]
         [ 4 5 6]]
        >>> output = F.narrow(x, 1, 1, 2)
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


@constexpr
def _raise_type_error():
    raise TypeError("The inputs type must be a Tensor, tuple or list of Tensors.")


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
        >>> from mindspore.ops.functional import vmap
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
string_eq = Primitive('string_equal')
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
mixed_precision_cast = Primitive("mixed_precision_cast")
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

tensor_operator_registry.register('addcdiv', P.Addcdiv)
tensor_operator_registry.register('addcmul', P.Addcmul)
tensor_operator_registry.register('all', P.ReduceAll)
tensor_operator_registry.register('any', P.ReduceAny)
tensor_operator_registry.register('atan2', atan2)
tensor_operator_registry.register('abs', P.Abs)
tensor_operator_registry.register('tan', P.Tan)
tensor_operator_registry.register('cosh', P.Cosh)
tensor_operator_registry.register('pow', P.Pow)
tensor_operator_registry.register('mean', P.ReduceMean)
tensor_operator_registry.register('prod', prod)
tensor_operator_registry.register('round', P.Round)
tensor_operator_registry.register('reshape', P.Reshape)
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
tensor_operator_registry.register('minimum', P.Minimum)
tensor_operator_registry.register('matrix_determinant', matrix_determinant)
tensor_operator_registry.register('log1p', log1p)
tensor_operator_registry.register('log_matrix_determinant', log_matrix_determinant)
tensor_operator_registry.register('ceil', P.Ceil)
tensor_operator_registry.register('fill', P.Fill)
tensor_operator_registry.register('tile', P.Tile)
tensor_operator_registry.register('logical_not', P.LogicalNot)
tensor_operator_registry.register('sum', P.ReduceSum)
tensor_operator_registry.register('split', P.Split)
tensor_operator_registry.register('select', P.Select)
tensor_operator_registry.register('zeros_like', P.ZerosLike)
tensor_operator_registry.register('scalar_to_tensor', scalar_to_tensor)
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
tensor_operator_registry.register('csr_mul', csr_mul)
tensor_operator_registry.register('csr2coo', csr2coo)
tensor_operator_registry.register('coo2csr', coo2csr)
tensor_operator_registry.register('csr_div', csr_div)
tensor_operator_registry.register('csr_mv', csr_mv)
tensor_operator_registry.register('csr_mm', _csr_ops.CSRMM)
tensor_operator_registry.register('csr_reduce_sum', csr_reduce_sum)
tensor_operator_registry.register('dense_to_sparse_csr', dense_to_sparse_csr)
tensor_operator_registry.register('dense_to_sparse_coo', dense_to_sparse_coo)
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
tensor_operator_registry.register('tensor_scatter_sub', P.TensorScatterSub)
tensor_operator_registry.register('tensor_scatter_add', P.TensorScatterAdd)
tensor_operator_registry.register('bernoulli', bernoulli)
tensor_operator_registry.register('norm', norm)
tensor_operator_registry.register('renorm', renorm)
tensor_operator_registry.register('adaptive_max_pool2d', AdaptiveMaxPool2D)
tensor_operator_registry.register('coalesce', coalesce)
tensor_operator_registry.register('arg_min_with_value', min)
tensor_operator_registry.register('unsorted_segment_sum', P.UnsortedSegmentSum)
tensor_operator_registry.register('coo_add', sparse_add)
__all__ = [name for name in dir() if name[0] != "_"]
__all__.remove('Primitive')
