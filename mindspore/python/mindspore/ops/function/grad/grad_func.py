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
import numpy as np
from mindspore.common import jit
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.nn.grad.cell_grad import _LinearizeInner
from mindspore.ops.primitive import constexpr
from mindspore.ops.function.array_func import ones, expand_dims, size, reshape, broadcast_to, transpose
from mindspore.ops.composite import _Vmap, _Grad, _TaylorOperation, GradOperation
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner

cast = P.Cast()
dtype = P.DType()
zeros = P.Zeros()
oneslike = P.OnesLike()


@constexpr
def _check_has_aux_type(inputs):
    if not isinstance(inputs, bool):
        raise TypeError("The 'has_aux' must be bool type.")
    return True


@constexpr
def _raise_type_error():
    raise TypeError("The inputs type must be a Tensor, tuple or list of Tensors.")


@constexpr
def _check_duplicate_grad_position(grad_position):
    """Check if `grad_position` has duplicate positions when `grad_position` has more than one numbers."""
    if len(set(grad_position)) != len(grad_position):
        raise ValueError("There are duplicate positions in `grad_position`, please check it")


@constexpr
def _convert_grad_position_type(grad_position):
    """Check and convert the type and size of grad position index."""
    if isinstance(grad_position, tuple):
        _check_duplicate_grad_position(grad_position)
        _grad_position = list(grad_position)
        for i, gp in enumerate(_grad_position):
            if isinstance(gp, bool):
                _grad_position[i] = int(gp)
            if not isinstance(gp, int):
                raise TypeError(f"For 'F.grad', the element in 'grad_position' must be int.")
            if gp < 0:
                raise ValueError("The element in grad_position must be >= 0.")
        grad_position = tuple(_grad_position)
    elif isinstance(grad_position, int):
        if grad_position < 0:
            raise ValueError("grad_position must be >= 0.")
        grad_position = (grad_position,)
    else:
        raise TypeError(f"For 'F.grad', the 'grad_position' must be int or tuple.")
    return grad_position


@constexpr
def _check_grad_position(grad_position, args_num):
    """Check and convert grad position index."""
    grad_position = _convert_grad_position_type(grad_position)
    for gp in grad_position:
        if gp < 0 or gp >= args_num:
            raise ValueError("The element in grad_position must belong to [0, args_num).")
    return grad_position


@constexpr
def _get_grad_op(get_by_list, get_by_position, has_aux, get_value=False, return_ids=False):
    return _Grad(get_by_list=get_by_list, get_by_position=get_by_position, has_aux=has_aux, get_value=get_value,
                 return_ids=return_ids)


def grad(fn, grad_position=0, weights=None, has_aux=False, return_ids=False):
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
        return_ids(bool): Whether return the tuple made by gradients and the index to specify which inputs
            to be differentiated or the name of parameters of the training network that need to calculate the gradient.
            If True, the output gradients will be replaced by the tuples made by gradients and the index to specify
            which inputs to be differentiated or the name of parameters of the training network.
            Default: False.

    Returns:
        Function, the gradient function to calculate gradient for the input function or cell.
        For example, as for `out1, out2 = fn(*args)`, when `has_aux` is set True, gradient function will return outputs
        like `(gradient, out2)` and `out2` does not contribute to the differentiation, otherwise `gradient`.
        When return_ids is set to True, The format of the output will be the same with the output of grad when
        return_ids is set to false, but every gradient in the output will be replaced by a tuple of position id or
        parameter name and its gradient.

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
        >>> from mindspore import grad
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
        >>> # For given network to be differentiated with both inputs and weights, there are 4 cases.
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
        >>> # Case 4: return the gradient with ids.
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, ops
        >>> from mindspore import grad
        >>>
        >>> # Cell object to be differentiated
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x * y * z
        >>> x = Tensor([1, 2], mindspore.float32)
        >>> y = Tensor([-2, 3], mindspore.float32)
        >>> z = Tensor([0, 3], mindspore.float32)
        >>> net = Net()
        >>> output = grad(net, grad_position=(1, 2), return_ids = True)(x, y, z)
        >>> print(output)
        ((1, Tensor(shape=[2], dtype=Float32, value=[ 0.00000000e+00,  6.00000000e+00])),
         (2, Tensor(shape=[2], dtype=Float32, value=[-2.00000000e+00,  6.00000000e+00])))
    """
    if grad_position is None and weights is None:
        raise ValueError("`grad_position` and `weight` can not be None at the same time.")

    if grad_position is None:
        return _get_grad_op(True, False, has_aux, False, return_ids)(fn, weights)

    grad_position = _convert_grad_position_type(grad_position)
    if weights is None:
        return _get_grad_op(False, True, has_aux, False, return_ids)(fn, None, grad_position)
    return _get_grad_op(True, True, has_aux, False, return_ids)(fn, weights, grad_position)


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
        >>> from mindspore import value_and_grad
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


def get_grad(gradients, identifier):
    """
    A function to get get expected gradient from the return value of ops.grad, when it has return_ids parameter set
    to True, by using the position id of a tensor or the parameter.

    As for gradient, three typical cases are included:

    1. gradient with respect to inputs. In this case, use return value of ops.grad as the first input and
       the position of the tensor as the second input.
    2. gradient with respect to weights. In this case, use return value of ops.grad as the first input and
       the parameter as the second input.

    Args:
        gradients (Union[tuple[int, Tensor], tuple[tuple, tuple]]): The return value of mindspore.grad when return_ids
            is set to True.
        identifier (Union[int, Parameter]): The position number of a tensor, or a parameter that is used in
            mindspore.grad.

    Returns:
        The gradient of the tensor on the position of the position number used as the second input, or the gradient
        of the parameter used as the second input.

    Raises:
        RuntimeError: If gradient is not found.
        TypeError: If type of Args does not belong to required ones.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, ops
        >>> from mindspore import grad, get_grad
        >>>
        >>>  # Cell object to be differentiated
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x * y * z
        >>> x = Tensor([1, 2], mindspore.float32)
        >>> y = Tensor([-2, 3], mindspore.float32)
        >>> z = Tensor([0, 3], mindspore.float32)
        >>> net = Net()
        >>> out_grad = grad(net, grad_position=(1, 2), return_ids=True)(x, y, z)
        >>> output = get_grad(out_grad, 1)
        >>> print(output)
        Tensor(shape=[2], dtype=Float32, value=[0.00000000e+00,  6.00000000e+00]
    """
    return inner.GetGrad()(gradients, identifier)


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
        >>> out_primals, out_series = ops.jet(net, primals, series)
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
        >>> out_primals, out_series = ops.derivative(net, primals, order)
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


_grad_single = GradOperation(sens_param=True)
_grad_all = GradOperation(sens_param=True, get_all=True)


def jvp(fn, inputs, v, has_aux=False):
    """
    Compute the jacobian-vector-product of the given network. `jvp` matches
    `forward-mode differentiation <https://www.mindspore.cn/docs/en/master/design/auto_gradient.html#forward-mode-ad>`_.

    Args:
        fn (Union[Function, Cell]): The function or net that takes Tensor inputs and returns single Tensor or tuple of
            Tensors.
        inputs (Union[Tensor, tuple[Tensor], list[Tensor]]): The inputs to `fn` .
        v (Union[Tensor, tuple[Tensor], list[Tensor]]): The vector in jacobian-vector-product. The shape and type of `v`
            should be the same as `inputs` .
        has_aux (bool): If True, only the first output of `fn` contributes the gradient of `fn`, while the other outputs
            will be returned straightly. It means the `fn` must return more than one outputs in this case.
            Default: False.

    Returns:
        - **net_output** (Union[Tensor, tuple[Tensor]]) - The output of `fn(inputs)` . Specially, when `has_aux` is set
          True, `netout` is the first output of `fn(inputs)` .
        - **jvp** (Union[Tensor, tuple[Tensor]]) - The result of jacobian-vector-product.
        - **aux_value** (Union[Tensor, tuple[Tensor]], optional) - When `has_aux` is True, `aux_value` will be returned.
          It means the second to last outputs of `fn(inputs)` . Specially, `aux_value` does not contribute to gradient.

    Raises:
        TypeError: `inputs` or `v` does not belong to required types.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import jvp
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = jvp(Net(), (x, y), (v, v))
        >>> print(output[0])
        [[ 2. 10.]
         [30. 68.]]
        >>> print(output[1])
        [[ 4. 13.]
         [28. 49.]]
        >>>
        >>> def fn(x, y):
        ...     return x ** 3 + y, y
        >>> output, jvp_out, aux = jvp(fn, (x, y), (v, v), has_aux=True)
        >>> print(output)
        [[ 2. 10.]
         [30. 68.]]
        >>> print(jvp_out)
        [[ 4. 13.]
         [28. 49.]]
        >>> print(aux)
        [[ 1. 2.]
         [3. 4.]]
    """
    _check_has_aux_type(has_aux)

    def aux_fn(*args):
        outputs = fn(*args)
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise ValueError("When 'has_aux' is True, origin 'fn' requires more than one outputs.")
        res = outputs[0]
        return res

    def grad_single(u, first_grad_single_value):
        if has_aux:
            return _grad_single(aux_fn)(*first_grad_single_value, u)
        return _grad_single(fn)(*first_grad_single_value, u)

    def grad_all(u, first_grad):
        if has_aux:
            return _grad_all(aux_fn)(*first_grad, u)
        return _grad_all(fn)(*first_grad, u)

    def _wrap_container_inner(*arg):
        jvp_inputs = arg[1:]
        vectors = arg[0]
        if has_aux:
            outputs = aux_fn(*jvp_inputs)
        else:
            outputs = fn(*jvp_inputs)
        if isinstance(outputs, tuple):
            u = ()
            for item in outputs:
                u = u + (oneslike(item),)
        else:
            u = oneslike(outputs)
        if len(jvp_inputs) == 1:
            second_grad_net = _grad_single(grad_single)
            gradient_outputs = second_grad_net(u, jvp_inputs, vectors)
        else:
            second_grad_net = _grad_single(grad_all)
            gradient_outputs = second_grad_net(u, jvp_inputs, vectors)
        if has_aux:
            res = fn(*jvp_inputs)
            if len(res) == 2:
                return res[0], gradient_outputs, res[1]
            return res[0], gradient_outputs, res[1:]
        return outputs, gradient_outputs

    if has_aux:
        @jit(hash_args=aux_fn)
        def _wrap_container(*arg):
            return _wrap_container_inner(*arg)
    else:
        @jit(hash_args=fn)
        def _wrap_container(*arg):
            return _wrap_container_inner(*arg)

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

    @jit(hash_args=fn)
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


def _check_tensor(inputs):
    if not isinstance(inputs, (Tensor, tuple)):
        raise TypeError("The inputs type must be Tensor.")
    if isinstance(inputs, tuple):
        for item in inputs:
            if not isinstance(item, (Tensor, tuple, list)):
                raise TypeError("The inputs type must be Tensor.")
    return True


def vjp(fn, *inputs, has_aux=False):
    """
    Compute the vector-jacobian-product of the given network. `vjp` matches
    `reverse-mode differentiation <https://www.mindspore.cn/docs/en/master/design/auto_gradient.html#reverse-mode-ad>`_.

    Args:
        fn (Union[Function, Cell]): The function or net that takes Tensor inputs and returns single Tensor or tuple of
            Tensors.
        inputs (Union[Tensor, tuple[Tensor], list[Tensor]]): The inputs to `fn` .
        has_aux (bool): If True, only the first output of `fn` contributes the gradient of `fn`, while the other outputs
            will be returned straightly. It means the `fn` must return more than one outputs in this case.
            Default: False.

    Returns:
        Forward outputs and function to calculate vjp.

        - **net_output** (Union[Tensor, tuple[Tensor]]) - The output of `fn(inputs)`. Specially, when `has_aux` is set
          True, `netout` is the first output of `fn(inputs)`.
        - **vjp_fn** (Function) - To calculate vector-jacobian-product. Its inputs are the vectors whose shape and
          type should be the same as `netout` .
        - **aux_value** (Union[Tensor, tuple[Tensor]], optional) - When `has_aux` is True, `aux_value` will be returned.
          It means the second to last outputs of `fn(inputs)`. Specially, `aux_value` does not contribute to gradient.

    Raises:
        TypeError: `inputs` or `v` does not belong to required types.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import vjp
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> outputs, vjp_fn = vjp(Net(), x, y)
        >>> print(outputs)
        [[ 2. 10.]
         [30. 68.]]
        >>> gradient = vjp_fn(v)
        >>> print(gradient)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 3.00000000e+00,  1.20000000e+01],
         [ 2.70000000e+01,  4.80000000e+01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.00000000e+00,  1.00000000e+00],
         [ 1.00000000e+00,  1.00000000e+00]]))
        >>> def fn(x, y):
        ...     return 2 * x + y, y ** 3
        >>> outputs, vjp_fn, aux = vjp(fn, x, y, has_aux=True)
        >>> gradient = vjp_fn(v)
        >>> print(outputs)
        [[ 3.  6.]
         [ 9. 12.]]
        >>> print(aux)
        [[ 1.  8.]
         [27. 64.]]
        >>> print(gradient)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 2.00000000e+00,  2.00000000e+00],
         [ 2.00000000e+00,  2.00000000e+00]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.00000000e+00,  1.00000000e+00],
         [ 1.00000000e+00,  1.00000000e+00]]))
    """
    _check_tensor(inputs)
    _check_has_aux_type(has_aux)

    def aux_fn(*args):
        outputs = fn(*args)
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise ValueError("When 'has_aux' is True, origin 'fn' requires more than one outputs.")
        res = outputs[0]
        return res

    def wrap_container(*v):
        _check_tensor(v)
        if has_aux:
            fn_ = aux_fn
        else:
            fn_ = fn
        if len(v) == 1:
            return _grad_all(fn_)(*inputs, v[0])
        return _grad_all(fn_)(*inputs, v)

    res = fn(*inputs)
    if has_aux:
        if len(res) == 2:
            return res[0], wrap_container, res[1]
        return res[0], wrap_container, res[1:]
    return res, wrap_container


@constexpr
def _jac_generate_target_dimension(x):
    """For given length = len(x), this method generates target dimension tuple (1, 2, 3,..., length, 0)."""
    target_dimension = tuple(index + 1 for index, _ in enumerate(x[1:])) + (0,)
    return target_dimension


def _jacfwd_trans_item(item, inputs_shape, grad_position):
    """transfer origin item to derivative of each output with respect to each input."""
    output_wrt_input_all = ()
    for i in grad_position:
        origin_output_wrt_input = item[inputs_shape[i][1]:inputs_shape[i + 1][1]]
        target_dimension = _jac_generate_target_dimension(origin_output_wrt_input.shape)
        temp = transpose(origin_output_wrt_input, target_dimension)
        output_wrt_input = reshape(temp, temp.shape[:-1] + inputs_shape[i + 1][0])
        output_wrt_input_all += (output_wrt_input,)
    return output_wrt_input_all


def _jacfwd_postprocess(x, inputs_shape, grad_position):
    """reformat jacobian."""
    if isinstance(x, tuple):
        jacobian = ()
        for item in x:
            jacobian += _jacfwd_trans_item(item, inputs_shape, grad_position)
        res = jacobian
    else:
        res = _jacfwd_trans_item(x, inputs_shape, grad_position)
    if len(res) == 1:
        return res[0]
    input_num = len(grad_position)
    if len(res) % input_num != 0:
        raise ValueError("The numbers of inputs and outputs do not match.")
    output_num = len(res) // input_num
    if input_num == 1 or output_num == 1:
        return res
    jac = ()
    for i in range(output_num):
        input_grad = ()
        for j in range(input_num):
            input_grad += (res[i * input_num + j],)
        jac += (input_grad,)
    return jac


def _jacfwd_construct_v(inputs, grad_position):
    """
    For input (x1, x2), x1.shape = (a, b), x2.shape = (c, d), this method generates corresponding v (v1, v2),
    v1.shape = (N, a, b), v2.shape = (N, c, d), while N = a*b + c*d.
    """
    v = ()
    primals = ()
    inputs_shape = (((), 0),)
    num = 0
    items_num = ()
    cum_num = (0,)
    for item in inputs:
        item_num = size(item)
        num += item_num
        inputs_shape += ((item.shape, num),)
        items_num += (item_num,)
        cum_num += (num,)
    for i, element in enumerate(inputs):
        item_size = items_num[i]
        if i in grad_position:
            temp2 = Tensor(np.eye(num, item_size, -cum_num[i], np.float32))
        else:
            temp2 = zeros((num, item_size), mstype.float32)
        input_v = reshape(temp2, (num,) + element.shape)
        primal = broadcast_to(element, (num,) + element.shape)
        v += (input_v,)
        primals += (primal,)
    if len(inputs) == 1:
        return primals, v[0], inputs_shape
    return primals, v, inputs_shape


_vmap = _Vmap()


def jacfwd(fn, grad_position=0, has_aux=False):
    """
    Compute Jacobian via forward mode, corresponding to
    `forward-mode differentiation <https://www.mindspore.cn/docs/en/master/design/auto_gradient.html#forward-mode-ad>`_.
    When number of outputs is much greater than that of inputs, it's better to calculate Jacobian via forward mode than
    reverse mode to get better performance.

    Args:
        fn (Union[Cell, Function]): Function to do GradOperation.
        grad_position (Union[int, tuple[int]]): If int, get the gradient with respect to single input.
            If tuple, get the gradients with respect to selected inputs. 'grad_position' begins with 0. Default: 0.
        has_aux (bool): If True, only the first output of `fn` contributes the gradient of `fn`, while the other outputs
            will be returned straightly. It means the `fn` must return more than one outputs in this case.
            Default: False.

    Returns:
        Function, returns the Jacobian function for the input function or cell.
        For example, as for `out1, out2 = fn(*args)`, when `has_aux` is set True, gradient function will return outputs
        like `(Jacobian, out2)` and `out2` does not contribute to the differentiation, otherwise `Jacobian` .

    Raises:
        TypeError: `grad_position` or `has_aux` does not belong to required types.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import jacfwd
        >>> from mindspore import Tensor
        >>> class MultipleInputsMultipleOutputsNet(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x ** 2 + y ** 2 + z ** 2, x * y * z
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> z = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> net = MultipleInputsMultipleOutputsNet()
        >>> jac, aux = jacfwd(net, grad_position=0, has_aux=True)(x, y, z)
        >>> print(jac)
        [[[[ 2.,  0.]
           [ 0.,  0.]]
          [[ 0.,  4.]
           [ 0.,  0.]]]
         [[[ 0.,  0.]
           [ 6.,  0.]]
          [[ 0.,  0.]
           [ 0.,  8.]]]]
        >>> print(aux)
        [[ 1.  4.]
         [ 9. 16.]]
    """
    _check_has_aux_type(has_aux)

    def aux_fn(*args):
        outputs = fn(*args)
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise ValueError("When 'has_aux' is True, origin 'fn' requires more than one outputs.")
        res = outputs[0]
        return res

    def grad_single(u, first_grad_single_value):
        if has_aux:
            return _grad_single(aux_fn)(*first_grad_single_value, u)
        return _grad_single(fn)(*first_grad_single_value, u)

    def grad_all(u, first_grad):
        if has_aux:
            return _grad_all(aux_fn)(*first_grad, u)
        return _grad_all(fn)(*first_grad, u)

    @jit
    def wrapped(*args):
        checked_grad_position = _check_grad_position(grad_position, len(args))
        primals, v, inputs_shape = _jacfwd_construct_v(args, checked_grad_position)

        def inner_fn(jvp_inputs, vectors):
            outputs = fn(*jvp_inputs)
            if isinstance(outputs, tuple):
                u = ()
                for item in outputs:
                    u = u + (oneslike(item),)
            else:
                u = oneslike(outputs)
            if len(jvp_inputs) == 1:
                second_grad_net = _grad_single(grad_single)
            else:
                second_grad_net = _grad_single(grad_all)
            gradient_outputs = second_grad_net(u, jvp_inputs, vectors)
            return gradient_outputs

        def inner_aux_fn(jvp_inputs, vectors):
            outputs = aux_fn(*jvp_inputs)
            u = oneslike(outputs)
            if len(jvp_inputs) == 1:
                second_grad_net = _grad_single(grad_single)
            else:
                second_grad_net = _grad_single(grad_all)
            gradient_outputs = second_grad_net(u, jvp_inputs, vectors)
            return gradient_outputs

        if has_aux:
            res = _vmap(inner_aux_fn)(primals, v)
            jac_res = _jacfwd_postprocess(res, inputs_shape, checked_grad_position)
            forward_outputs = fn(*args)
            if len(forward_outputs) == 2:
                return jac_res, forward_outputs[1]
            return jac_res, forward_outputs[1:]
        res = _vmap(inner_fn)(primals, v)
        jac_res = _jacfwd_postprocess(res, inputs_shape, checked_grad_position)
        return jac_res

    return wrapped


def _jacrev_trans_item(item, outputs_shape):
    """transfer origin item to derivative of each output with respect to each input."""
    output_wrt_input_all = ()
    length = len(outputs_shape) - 1
    for i in range(length):
        origin_output_wrt_input = item[outputs_shape[i][1]:outputs_shape[i + 1][1]]
        target_dimension = _jac_generate_target_dimension(origin_output_wrt_input.shape)
        temp = transpose(origin_output_wrt_input, target_dimension)
        output_wrt_input = reshape(origin_output_wrt_input, outputs_shape[i + 1][0] + temp.shape[:-1])
        output_wrt_input_all += (output_wrt_input,)
    return output_wrt_input_all


def _jacrev_postprocess(x, outputs_shape, grad_position):
    """reformat jacobian."""
    if isinstance(x, tuple):
        jacobian = ()
        for item in x:
            jacobian += _jacrev_trans_item(item, outputs_shape)
        res = jacobian
    else:
        res = _jacrev_trans_item(x, outputs_shape)
    if len(res) == 1:
        return res[0]
    input_num = len(grad_position)
    if len(res) % input_num != 0:
        raise ValueError("The numbers of inputs and outputs do not match.")
    output_num = len(res) // input_num
    if input_num == 1 or output_num == 1:
        return res
    jac = ()
    for i in range(output_num):
        input_grad = ()
        for j in range(input_num):
            input_grad += (res[j * output_num + i],)
        jac += (input_grad,)
    return jac


def _jacrev_construct_v(inputs, outputs, has_aux=False):
    """
    For outputs (y1, y2), y1.shape = (a, b), y2.shape = (c, d), this method generates corresponding v (v1, v2),
    v1.shape = (N, a, b), v2.shape = (N, c, d), while N = a*b + c*d.
    """
    if isinstance(outputs, Tensor):
        outputs = (outputs,)
    if has_aux:
        outputs = (outputs[0],)
    v = ()
    primals = ()
    outputs_shape = (((), 0),)
    num = 0
    items_num = ()
    cum_num = (0,)
    for item in outputs:
        item_num = size(item)
        num += item_num
        outputs_shape += ((item.shape, num),)
        items_num += (item_num,)
        cum_num += (num,)
    for i, element in enumerate(inputs):
        primal = broadcast_to(element, (num,) + element.shape)
        primals += (primal,)
    for i, element in enumerate(outputs):
        item_size = items_num[i]
        temp2 = Tensor(np.eye(num, item_size, -cum_num[i], np.float32))
        output_v = reshape(temp2, (num,) + element.shape)
        v += (output_v,)
    if len(outputs) == 1 or has_aux:
        return primals, v[0], outputs_shape
    return primals, v, outputs_shape


_grad = _Grad(get_by_position=True, has_aux=False, sens_param=True)


def jacrev(fn, grad_position=0, has_aux=False):
    """
    Compute Jacobian via reverse mode, corresponding to
    `reverse-mode differentiation <https://www.mindspore.cn/docs/en/master/design/auto_gradient.html#reverse-mode-ad>`_.
    When number of inputs is much greater than that of outputs, it's better to calculate Jacobian via reverse mode than
    forward mode to get better performance.

    Args:
        fn (Union[Cell, Function]): Function to do GradOperation.
        grad_position (Union[int, tuple[int]]): If int, get the gradient with respect to single input.
            If tuple, get the gradients with respect to selected inputs. 'grad_position' begins with 0. Default: 0.
        has_aux (bool): If True, only the first output of `fn` contributes the gradient of `fn`, while the other outputs
            will be returned straightly. It means the `fn` must return more than one outputs in this case.
            Default: False.

    Returns:
        Function, returns the Jacobian function for the input function or cell.
        For example, as for `out1, out2 = fn(*args)`, when `has_aux` is set True, gradient function will return outputs
        like `(Jacobian, out2)` and `out2` does not contribute to the differentiation, otherwise `Jacobian` .

    Raises:
        TypeError: `grad_position` or `has_aux` does not belong to required types.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import jacrev
        >>> from mindspore import Tensor
        >>> class MultipleInputsMultipleOutputsNet(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x ** 2 + y ** 2 + z ** 2, x * y * z
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> z = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> net = MultipleInputsMultipleOutputsNet()
        >>> jac, aux = jacrev(net, grad_position=0, has_aux=True)(x, y, z)
        >>> print(jac)
        [[[[ 2.,  0.]
           [ 0.,  0.]]
          [[ 0.,  4.]
           [ 0.,  0.]]]
         [[[ 0.,  0.]
           [ 6.,  0.]]
          [[ 0.,  0.]
           [ 0.,  8.]]]]
        >>> print(aux)
        [[ 1.  4.]
         [ 9. 16.]]
    """
    _check_has_aux_type(has_aux)

    def aux_fn(*args):
        outputs = fn(*args)
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise ValueError("When 'has_aux' is True, origin 'fn' requires more than one outputs.")
        res = outputs[0]
        return res

    @jit
    def wrapped(*args):
        checked_grad_position = _check_grad_position(grad_position, len(args))
        outputs = fn(*args)
        primals, v, outputs_shape = _jacrev_construct_v(args, outputs, has_aux)

        def inner_fn(vjp_inputs, vectors):
            gradient_outputs = _grad(fn, None, checked_grad_position)(*vjp_inputs, vectors)
            return gradient_outputs

        def inner_aux_fn(vjp_inputs, vectors):
            gradient_outputs = _grad(aux_fn, None, checked_grad_position)(*vjp_inputs, vectors)
            return gradient_outputs

        if has_aux:
            res = _vmap(inner_aux_fn)(primals, v)
            jac_res = _jacrev_postprocess(res, outputs_shape, checked_grad_position)
            forward_outputs = fn(*args)
            if len(forward_outputs) == 2:
                return jac_res, forward_outputs[1]
            return jac_res, forward_outputs[1:]

        res = _vmap(inner_fn)(primals, v)
        jac_res = _jacrev_postprocess(res, outputs_shape, checked_grad_position)
        return jac_res

    return wrapped


def custom_vjp(fn=None):
    """
    Support vjp to custom bprop for function.

    Args:
        fn (function): The `fn` that need to define custom bprop. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def deco(fn):
        class CustomVjp(Cell):
            """
            The CustomVjp decorates function into cell to support custom bprop.
            """

            def __init__(self, fwd):
                super(CustomVjp, self).__init__()
                self.fwd = fwd
                self.bwd = None
                self.add_flags(custom_vjp=True)

            def construct(self, *args):
                return self.fwd(*args)

            def defbwd(self, bwd):
                self.bwd = bwd

            def bprop(self, *args):
                return self.bwd(*args)

        return CustomVjp(fn)

    if fn is not None:
        return deco(fn)
    return deco


def stop_gradient(value):
    """
    StopGradient is used for eliminating the effect of a value on the gradient, such as truncating
    the gradient propagation from an output of a function.
    For more details, please refer to `Stop Gradient
    <https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html#stop-gradient>`_.

    Args:
        value (Any): The value whose effect on the gradient to be eliminated.

    Returns:
        The same as `value`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> def net(x, y):
        ...     out1 = ops.MatMul()(x, y)
        ...     out2 = ops.MatMul()(x, y)
        ...     out2 = ops.stop_gradient(out2)
        ...     return out1, out2
        ...
        >>> x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
        >>> y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
        >>> grad_fn = ops.grad(net)
        >>> output = grad_fn(x, y)
        >>> print(output)
        [[1.4100001 1.6       6.5999994]
         [1.4100001 1.6       6.5999994]]
    """
    return P.StopGradient()(value)


__all__ = [
    'grad',
    'value_and_grad',
    'jacfwd',
    'jacrev',
    'jet',
    'derivative',
    'jvp',
    'vjp',
    'linearize',
    'stop_gradient',
    'get_grad'
]
__all__.sort()
