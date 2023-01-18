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

"""Defines gradient related operators with functional form."""

from __future__ import absolute_import
from functools import partial
from mindspore.common import ms_function
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.grad.cell_grad import _JvpInner
from mindspore.nn.grad.cell_grad import _VjpInner
from mindspore.nn.grad.cell_grad import _LinearizeInner
from mindspore.ops.primitive import constexpr
from mindspore.ops.function import ones, expand_dims
from mindspore.ops.composite import _Grad, _TaylorOperation
from mindspore.ops import operations as P

cast = P.Cast()
dtype = P.DType()
zeros = P.Zeros()


@constexpr
def _raise_type_error():
    raise TypeError("The inputs type must be a Tensor, tuple or list of Tensors.")


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


@constexpr
def _get_grad_op(get_by_list, get_by_position, has_aux, get_value=False):
    return _Grad(get_by_list=get_by_list, get_by_position=get_by_position, has_aux=has_aux, get_value=get_value)


def grad(fn, grad_position=0, weights=None, has_aux=False):
    """
    A wrapper function to generate the gradient function for the input function.

    As for gradient, three typical cases are included:

    1. gradient with respect to inputs. In this case, `grad_position` is not None while `weights` is None.
    2. gradient with respect to weights. In this case, `grad_position` is None while `weights` is not None.
    3. gradient with respect to inputs and weights. In this case, `grad_position` and `weights` are not None.

    Args:
        fn (Union[Cell, Function]): Function to do GradOperation.
        grad_position (Union[NoneType, int, tuple[int]]): Index to specify which inputs to be differentiated.
            If int, get the gradient with respect to single input.
            If tuple, get the gradients with respect to selected inputs. `grad_position` begins with 0.
            If None, none derivative of any input will be figured out, and in this case, `weights` is required.
            Default: 0.
        weights (Union[ParameterTuple, Parameter, list[Parameter]]): The parameters of the training network that need to
            calculate the gradient. `weights` can be got through `weights = net.trainable_params()` .
            Default: None.
        has_aux (bool): If True, only the first output of `fn` contributes the gradient of `fn`, while the other outputs
            will be returned straightly. It means the `fn` must return more than one outputs in this case.
            Default: False.

    Returns:
        Function, the gradient function to calculate gradient for the input function or cell.
        For example, as for `out1, out2 = fn(*args)`, when `has_aux` is set True, gradient function will return outputs
        like `(gradient, out2)` and `out2` does not contribute to the differentiation, otherwise `gradient`.

    Raises:
        ValueError: If both `grad_position` and `weights` are None.
        TypeError: If type of Args does not belong to required ones.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, ops
        >>> from mindspore.ops import grad
        >>>
        >>> # Cell object to be differentiated
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x * y * z
        >>> x = Tensor([1, 2], mindspore.float32)
        >>> y = Tensor([-2, 3], mindspore.float32)
        >>> z = Tensor([0, 3], mindspore.float32)
        >>> net = Net()
        >>> output = grad(net, grad_position=(1, 2))(x, y, z)
        >>> print(output)
        (Tensor(shape=[2], dtype=Float32, value=[ 0.00000000e+00,  6.00000000e+00]),
         Tensor(shape=[2], dtype=Float32, value=[-2.00000000e+00,  6.00000000e+00]))
        >>>
        >>> # Function object to be differentiated
        >>> def fn(x, y, z):
        ...     res = x * ops.exp(y) * ops.pow(z, 2)
        ...     return res, z
        >>> x = Tensor([3, 3], mindspore.float32)
        >>> y = Tensor([0, 0], mindspore.float32)
        >>> z = Tensor([5, 5], mindspore.float32)
        >>> gradient, aux = grad(fn, (1, 2), None, True)(x, y, z)
        >>> print(gradient)
        (Tensor(shape=[2], dtype=Float32, value= [ 7.50000000e+01,  7.50000000e+01]),
         Tensor(shape=[2], dtype=Float32, value= [ 3.00000000e+01,  3.00000000e+01]))
        >>> print(aux)
        (Tensor(shape=[2], dtype=Float32, value= [ 5.00000000e+00,  5.00000000e+00]),)
        >>>
        >>> # For given network to be differentiated with both inputs and weights, there are 3 cases.
        >>> net = nn.Dense(10, 1)
        >>> loss_fn = nn.MSELoss()
        >>> def forward(inputs, labels):
        ...     logits = net(inputs)
        ...     loss = loss_fn(logits, labels)
        ...     return loss, logits
        >>> inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
        >>> labels = Tensor(np.random.randn(16, 1).astype(np.float32))
        >>> weights = net.trainable_params()
        >>> # Case 1: gradient with respect to inputs.
        >>> # Aux value does not contribute to the gradient.
        >>> grad_fn = grad(forward, grad_position=(0, 1), weights=None, has_aux=True)
        >>> inputs_gradient, (aux_logits,) = grad_fn(inputs, labels)
        >>> print(len(inputs_gradient))
        2
        >>> print(aux_logits.shape)
        (16, 1)
        >>>
        >>> # Case 2: gradient with respect to weights.
        >>> grad_fn = grad(forward, grad_position=None, weights=weights, has_aux=True)
        >>> params_gradient, (aux_logits,) = grad_fn(inputs, labels)
        >>> print(len(weights), len(params_gradient))
        2 2
        >>> print(aux_logits.shape)
        (16, 1)
        >>>
        >>> # Case 3: gradient with respect to inputs and weights.
        >>> grad_fn = grad(forward, grad_position=0, weights=weights, has_aux=False)
        >>> inputs_gradient, params_gradient = grad_fn(inputs, labels)
        >>> print(len(weights), len(params_gradient))
        2 2
    """
    if grad_position is None and weights is None:
        raise ValueError("`grad_position` and `weight` can not be None at the same time.")

    if grad_position is None:
        return _get_grad_op(True, False, has_aux)(fn, weights)

    grad_position = _convert_grad_position_type(grad_position)
    if weights is None:
        return _get_grad_op(False, True, has_aux)(fn, None, grad_position)
    return _get_grad_op(True, True, has_aux)(fn, weights, grad_position)


def value_and_grad(fn, grad_position=0, weights=None, has_aux=False):
    """
    A wrapper function to generate the function to calculate forward output and gradient for the input function.

    As for gradient, three typical cases are included:

    1. gradient with respect to inputs. In this case, `grad_position` is not None while `weights` is None.
    2. gradient with respect to weights. In this case, `grad_position` is None while `weights` is not None.
    3. gradient with respect to inputs and weights. In this case, `grad_position` and `weights` are not None.

    Args:
        fn (Union[Cell, Function]): Function to do GradOperation.
        grad_position (Union[NoneType, int, tuple[int]]): Index to specify which inputs to be differentiated.
            If int, get the gradient with respect to single input.
            If tuple, get the gradients with respect to selected inputs. `grad_position` begins with 0.
            If None, none derivative of any input will be solved, and in this case, `weights` is required.
            Default: 0.
        weights (Union[ParameterTuple, Parameter, list[Parameter]]): The parameters of the training network that need to
            calculate the gradient. `weights` can be got through `weights = net.trainable_params()` .
            Default: None.
        has_aux (bool): If True, only the first output of `fn` contributes the gradient of `fn`, while the other outputs
            will be returned straightly. It means the `fn` must return more than one outputs in this case.
            Default: False.

    Returns:
        Function, returns the gradient function to calculate forward output and gradient for the input function or cell.
        For example, as for `out1, out2 = fn(*args)` , gradient function will return outputs like
        `((out1, out2), gradient)` . When `has_aux` is set True, only `out1` contributes to the differentiation.

    Raises:
        ValueError: If both `grad_position` and `weights` are None.
        TypeError: If type of Args does not belong to required ones.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops, nn
        >>> from mindspore.ops import value_and_grad
        >>>
        >>> # Cell object to be differentiated
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x * y * z
        >>> x = Tensor([1, 2], mindspore.float32)
        >>> y = Tensor([-2, 3], mindspore.float32)
        >>> z = Tensor([0, 3], mindspore.float32)
        >>> net = Net()
        >>> grad_fn = value_and_grad(net, grad_position=1)
        >>> output, inputs_gradient = grad_fn(x, y, z)
        >>> print(output)
        [-0. 18.]
        >>> print(inputs_gradient)
        [0. 6.]
        >>>
        >>> # Function object to be differentiated
        >>> def fn(x, y, z):
        ...     res = x * ops.exp(y) * ops.pow(z, 2)
        ...     return res, z
        >>> x = Tensor(np.array([3, 3]).astype(np.float32))
        >>> y = Tensor(np.array([0, 0]).astype(np.float32))
        >>> z = Tensor(np.array([5, 5]).astype(np.float32))
        >>> output, inputs_gradient = value_and_grad(fn, grad_position=(1, 2), weights=None, has_aux=True)(x, y, z)
        >>> print(output)
        (Tensor(shape=[2], dtype=Float32, value= [ 7.50000000e+01,  7.50000000e+01]),
         Tensor(shape=[2], dtype=Float32, value= [ 5.00000000e+00,  5.00000000e+00]))
        >>> print(inputs_gradient)
        (Tensor(shape=[2], dtype=Float32, value= [ 7.50000000e+01,  7.50000000e+01]),
         Tensor(shape=[2], dtype=Float32, value= [ 3.00000000e+01,  3.00000000e+01]))
        >>>
        >>> # For given network to be differentiated with both inputs and weights, there are 3 cases.
        >>> net = nn.Dense(10, 1)
        >>> loss_fn = nn.MSELoss()
        >>> def forward(inputs, labels):
        ...     logits = net(inputs)
        ...     loss = loss_fn(logits, labels)
        ...     return loss, logits
        >>> inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
        >>> labels = Tensor(np.random.randn(16, 1).astype(np.float32))
        >>> weights = net.trainable_params()
        >>>
        >>> # Case 1: gradient with respect to inputs.
        >>> # For has_aux is set True, only loss contributes to the gradient.
        >>> grad_fn = value_and_grad(forward, grad_position=0, weights=None, has_aux=True)
        >>> (loss, logits), inputs_gradient = grad_fn(inputs, labels)
        >>> print(logits.shape)
        (16, 1)
        >>> print(inputs.shape, inputs_gradient.shape)
        (16, 10) (16, 10)
        >>>
        >>> # Case 2: gradient with respect to weights.
        >>> # For has_aux is set True, only loss contributes to the gradient.
        >>> grad_fn = value_and_grad(forward, grad_position=None, weights=weights, has_aux=True)
        >>> (loss, logits), params_gradient = grad_fn(inputs, labels)
        >>> print(logits.shape)
        (16, 1)
        >>> print(len(weights), len(params_gradient))
        2 2
        >>>
        >>> # Case 3: gradient with respect to inputs and weights.
        >>> # For has_aux is set False, both loss and logits contribute to the gradient.
        >>> grad_fn = value_and_grad(forward, grad_position=0, weights=weights, has_aux=False)
        >>> (loss, logits), (inputs_gradient, params_gradient) = grad_fn(inputs, labels)
        >>> print(logits.shape)
        (16, 1)
        >>> print(inputs.shape, inputs_gradient.shape)
        (16, 10) (16, 10)
        >>> print(len(weights), len(params_gradient))
        2 2
    """
    if grad_position is None and weights is None:
        raise ValueError("`grad_position` and `weight` can not be None at the same time.")

    if grad_position is None:
        return _get_grad_op(True, False, has_aux, True)(fn, weights)

    grad_position = _convert_grad_position_type(grad_position)
    if weights is None:
        return _get_grad_op(False, True, has_aux, True)(fn, None, grad_position)
    return _get_grad_op(True, True, has_aux, True)(fn, weights, grad_position)


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

    Note:
        If `primals` is Tensor of int type, it will be converted to Tensor of float type.

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
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> from mindspore.ops import jet
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.sin = ops.Sin()
        ...         self.exp = ops.Exp()
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
    if isinstance(primals, list) and len(primals) > 1:
        inputs = map(_preprocess_jet, primals, series)
        outputs = derivative_fn(*inputs)
    else:
        inputs = _preprocess_jet(primals, series)
        outputs = derivative_fn(inputs)
    if isinstance(outputs, tuple) and len(outputs) > 1:
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

    Note:
        If `primals` is Tensor of int type, it will be converted to Tensor of float type.

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
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> from mindspore.ops import derivative
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.sin = ops.Sin()
        ...         self.exp = ops.Exp()
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
    if isinstance(outputs, tuple) and len(outputs) > 1:
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
    Compute the jacobian-vector-product of the given network. `jvp` matches
    `forward-mode differentiation <https://www.mindspore.cn/docs/en/r1.10/design/auto_gradient.html#forward-mode-ad>`_.

    Args:
        fn (Union[Function, Cell]): The function or net that takes Tensor inputs and returns single Tensor or tuple of
            Tensors.
        inputs (Union[Tensor, tuple[Tensor], list[Tensor]]): The inputs to `fn` .
        v (Union[Tensor, tuple[Tensor], list[Tensor]]): The vector in jacobian-vector-product. The shape and type of `v`
            should be the same as `inputs` .

    Returns:
        - **net_output** (Union[Tensor, tuple[Tensor]]) - The result of `fn(inputs)` .
        - **jvp** (Union[Tensor, tuple[Tensor]]) - The result of jacobian-vector-product.

    Raises:
        TypeError: `inputs` or `v` does not belong to required types.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = ops.jvp(Net(), (x, y), (v, v))
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


def linearize(fn, inputs):
    """
    Produces a linear approximation to fun using jvp() and partial eval.
    This function is mainly useful if you want to apply jvp multiple times.

    Args:
        fn (Union[Function, Cell]): The function or net that takes Tensor inputs and returns single tensor or tuple of
            Tensors.
        inputs (Union[Tensor, Tuple or List of Tensors]): The inputs to `fn`.

    Returns:
        Tuple, tuple of output and jvp_fn.

        - **netout** (Tensor or Tuple of Tensors) - The output of "fn(inputs)".
        - **jvp_fn** (Function) - The function that evaluates the Jacobian-vector product.

    Raises:
        TypeError: If the input is not a tensor or tuple or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, ops
        >>> from mindspore import nn
        >>> from mindspore.ops.functional import linearize

        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.matmul = ops.MatMul()
        ...     def construct(self, x, y):
        ...         out = self.matmul(x, y)
        ...         return out
        >>> x = Tensor(np.array([[1, 2, 3], [3, 4, 5]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32))
        >>> v = (Tensor(np.array([[1, 1, 1], [1, 1, 1]]).astype(np.float32)),
        ...      Tensor(np.array([[1, 1], [1, 1], [0, 0]]).astype(np.float32)))
        >>> output, jvp_fn = linearize(Net(), (x, y))
        >>> print(output)
        [[22. 28.]
         [40. 52.]]
        >>> jvp = jvp_fn(v)
        >>> print(jvp)
        [[12. 15.]
         [16. 19.]]
    """
    linearize_inner = _LinearizeInner()

    @ms_function(hash_args=fn)
    def _wrap_container(*arg):
        args = arg[1:-1]
        vectors = arg[-1]
        output = arg[0]
        if isinstance(vectors, list):
            vectors = tuple(vectors)
        return linearize_inner(fn, vectors, output, args)

    if not isinstance(inputs, (Tensor, tuple, list)):
        _raise_type_error()
    if isinstance(inputs, Tensor):
        inputs = (inputs,)
    output = fn(*inputs)
    return output, partial(_wrap_container, output, *inputs)


def vjp(fn, inputs, v):
    """
    Compute the vector-jacobian-product of the given network. `vjp` matches
    `reverse-mode differentiation <https://www.mindspore.cn/docs/en/r1.10/design/auto_gradient.html#reverse-mode-ad>`_.

    Note:
        This function is subjected to change in the future.

    Args:
        fn (Union[Function, Cell]): The function or net that takes Tensor inputs and returns single Tensor or tuple of
            Tensors.
        inputs (Union[Tensor, tuple[Tensor], list[Tensor]]): The inputs to `fn` .
        v (Union[Tensor, tuple[Tensor], list[Tensor]]): The vector in vector-jacobian-product. The shape and type of `v`
            should be the same as `fn(inputs)` .

    Returns:
        - **net_output** (Union[Tensor, tuple[Tensor]]) - The result of `fn(inputs)` .
        - **vjp** (Union[Tensor, tuple[Tensor]]) - The result of vector-jacobian-product.

    Raises:
        TypeError: `inputs` or `v` does not belong to required types.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = ops.vjp(Net(), (x, y), v)
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


__all__ = [
    'grad',
    'value_and_grad',
    'jet',
    'derivative',
    'jvp',
    'vjp',
    'linearize'
]
__all__.sort()
