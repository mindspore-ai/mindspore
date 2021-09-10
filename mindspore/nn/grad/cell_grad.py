# Copyright 2021 Huawei Technologies Co., Ltd
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
"""cell grad"""
from ..cell import Cell
from ...ops import composite as C
from ...ops import operations as P
from ...ops.primitive import Primitive
from ...common import dtype as mstype
from ...common.api import ms_function


class _FirstGrad(Cell):
    def __init__(self, fn):
        super(_FirstGrad, self).__init__()
        self.first_grad_op = C.GradOperation(sens_param=True, get_all=True)
        self.fn = fn

    def construct(self, u, first_grad_input):
        return self.first_grad_op(self.fn)(*first_grad_input, u)


class _FirstGradSingleValue(Cell):
    def __init__(self, fn):
        super(_FirstGradSingleValue, self).__init__()
        self.first_grad_single_value_op = C.GradOperation(sens_param=True)
        self.fn = fn

    def construct(self, u, first_grad_single_value_input):
        return self.first_grad_single_value_op(self.fn)(*first_grad_single_value_input, u)


class Jvp(Cell):
    """
    Compute the jacobian-vector-product of the given network. Jvp is equivalent to forward mode autodiff.

    Args:
        network (Cell): The network that takes Tensor inputs and returns a tuple of Tensors or a Tensor.

    Inputs:
        - **inputs** (Tensors) - The inputs to `net`.
        - **v** (Tensors or Tuple of Tensors) - The vector for which the Jacobian vector product is computed.
          Must have the same size as the input of `network`.

    Outputs:
        A tuple with 2 Tensors or Tuple of Tensors:
        - **net_output** (Tensors or Tuple of Tensors) - The output of `network(inputs)`.
        - **jvp** (Tensors or Tuple of Tensors) - The result of the jacobian vector product.

    Examples:
        >>> from mindspore.nn import Jvp
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = Jvp(Net())(x, y, (v, v))
    """
    def __init__(self, fn):
        super(Jvp, self).__init__()
        self.fn = fn
        self.oneslike = P.OnesLike()
        self.first_grad = _FirstGrad(fn)
        self.first_grad.add_flags(enable_tuple_grad=True)
        self.first_grad_single_value = _FirstGradSingleValue(fn)
        self.first_grad_single_value.add_flags(enable_tuple_grad=True)
        self.second_grad_op = C.GradOperation(sens_param=True)
        self.issubclass_ = P.IsSubClass()
        self.typeof = Primitive('typeof')
        self.make_tuple = Primitive('MakeTuple')
        self.tuple_len = Primitive("tuple_len")

    @ms_function
    def construct(self, *args):
        jvp_input = args[0:-1]
        v = args[-1]
        output = self.fn(*jvp_input)

        if self.issubclass_(self.typeof(output), mstype.tuple_):
            u = self.make_tuple()
            for i in range(self.tuple_len(output)):
                u = u + self.make_tuple(self.oneslike(output[i]))
        else:
            u = self.oneslike(output)

        if self.tuple_len(jvp_input) == 1:
            second_gradient_net = self.second_grad_op(self.first_grad_single_value)
            gradient_output = second_gradient_net(u, jvp_input, v)
        else:
            second_gradient_net = self.second_grad_op(self.first_grad)
            gradient_output = second_gradient_net(u, jvp_input, v)
        return output, gradient_output


class Vjp(Cell):
    """
    Computes the dot product between a vector `v` and the Jacobian of the given network at the point
    given by the inputs.

    Args:
        network (Cell): The network that takes Tensor inputs and returns a tuple of Tensors or a Tensor.

    Inputs:
        - **inputs** (Tensors) - The inputs to `net`. Must be a tuple or a list.
        - **v** (Tensors or Tuple of Tensors) - The vector for which the vector Jacobian product is computed.
          Must have the same size as the output of `network`.

    Outputs:
        A tuple with 2 Tensors or Tuple of Tensors:
        - **net_output** (Tensors or Tuple of Tensors) - The output of `network(inputs)`.
        - **vjp** (Tensors or Tuple of Tensors) - The result of the dot product.

    Examples:
        >>> from mindspore.nn import Vjp
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = Vjp(Net())(x, y, v)
    """

    def __init__(self, fn):
        super(Vjp, self).__init__()
        self.fn = fn
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.grad_single_value = C.GradOperation(sens_param=True)
        self.issubclass_ = P.IsSubClass()
        self.typeof = Primitive('typeof')
        self.tuple_len = Primitive("tuple_len")

    @ms_function
    def construct(self, *args):
        front_input = args[0:-1]
        output = self.fn(*front_input)
        if self.tuple_len(front_input) == 1:
            gradient_output = self.grad_single_value(self.fn)(*args)
        else:
            gradient_output = self.grad(self.fn)(*args)
        return output, gradient_output
