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
from __future__ import absolute_import

from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.primitive import Primitive
from mindspore.common import dtype as mstype
from mindspore.common.api import jit
from mindspore.common._decorator import deprecated
from mindspore.common import mutable


class _FirstGrad(Cell):
    def __init__(self, fn):
        super(_FirstGrad, self).__init__()
        self.first_grad_op = C.GradOperation(sens_param=True, get_all=True)
        self.fn = fn

    def construct(self, u, first_grad_input):
        return self.first_grad_op(self.fn)(*first_grad_input, u)


class _JvpFirstGrad(Cell):
    def __init__(self):
        super(_JvpFirstGrad, self).__init__()
        self.first_grad_op = C.GradOperation(sens_param=True, get_all=True)

    def construct(self, u, fn, first_grad_input):
        return self.first_grad_op(fn)(*first_grad_input, u)


class _FirstGradSingleValue(Cell):
    def __init__(self, fn):
        super(_FirstGradSingleValue, self).__init__()
        self.first_grad_single_value_op = C.GradOperation(sens_param=True)
        self.fn = fn

    def construct(self, u, first_grad_single_value_input):
        return self.first_grad_single_value_op(self.fn)(*first_grad_single_value_input, u)


class _JvpFirstGradSingleValue(Cell):
    def __init__(self):
        super(_JvpFirstGradSingleValue, self).__init__()
        self.first_grad_single_value_op = C.GradOperation(sens_param=True)

    def construct(self, u, fn, first_grad_single_value_input):
        return self.first_grad_single_value_op(fn)(*first_grad_single_value_input, u)


class Jvp(Cell):
    """
    Jvp will be deprecated in the future, please use :func:`mindspore.ops.jvp` instead.

    Supported Platforms:
        Deprecated
    """

    @deprecated("1.9", "ops.jvp", True)
    def __init__(self, fn):
        super(Jvp, self).__init__()
        self.fn = fn
        self.oneslike = P.OnesLike()
        self.first_grad = _FirstGrad(fn)
        self.first_grad.add_flags(enable_tuple_grad_first=True)
        self.first_grad_single_value = _FirstGradSingleValue(fn)
        self.first_grad_single_value.add_flags(enable_tuple_grad_first=True)
        self.second_grad_op = C.GradOperation(sens_param=True)
        self.issubclass_ = inner.IsSubClass()
        self.typeof = Primitive('typeof')
        self.make_tuple = Primitive('MakeTuple')

    @jit
    def construct(self, *args):
        """construct for jvp."""
        jvp_input = args[0:-1]
        v = args[-1]
        output = self.fn(*jvp_input)

        if self.issubclass_(self.typeof(output), mstype.tuple_):
            u = self.make_tuple()
            for _, element in enumerate(output):
                u = u + self.make_tuple(mutable(self.oneslike(element)))
        else:
            u = mutable(self.oneslike(output))

        if len(jvp_input) == 1:
            second_gradient_net = self.second_grad_op(self.first_grad_single_value)
            gradient_output = second_gradient_net(u, jvp_input, v)
        else:
            second_gradient_net = self.second_grad_op(self.first_grad)
            gradient_output = second_gradient_net(u, jvp_input, v)
        return output, gradient_output


class _JvpInner(Cell):
    """
    Compute the jacobian-vector-product of the given network. Jvp is equivalent to forward mode autodiff.
    This class implements the inner process of function jvp.
    """

    def __init__(self):
        super(_JvpInner, self).__init__()
        self.oneslike = P.OnesLike()
        self.first_grad = _JvpFirstGrad()
        self.first_grad.add_flags(enable_tuple_grad_first=True)
        self.first_grad_single_value = _JvpFirstGradSingleValue()
        self.first_grad_single_value.add_flags(enable_tuple_grad_first=True)
        self.second_grad_op = C.GradOperation(sens_param=True)
        self.issubclass_ = inner.IsSubClass()
        self.typeof = Primitive('typeof')
        self.make_tuple = Primitive('MakeTuple')

    def compute_jvp(self, fn, v, jvp_input, output):
        """Compute the jacobian-vector-product of the given fn, vector, inputs and outputs."""
        if self.issubclass_(self.typeof(output), mstype.tuple_):
            u = self.make_tuple()
            for _, element in enumerate(output):
                u = u + self.make_tuple(mutable(self.oneslike(element)))
        else:
            u = mutable(self.oneslike(output))

        if len(jvp_input) == 1:
            second_gradient_net = self.second_grad_op(self.first_grad_single_value)
            gradient_output = second_gradient_net(u, fn, jvp_input, v)
        else:
            second_gradient_net = self.second_grad_op(self.first_grad)
            gradient_output = second_gradient_net(u, fn, jvp_input, v)
        return gradient_output

    def construct(self, *args):
        fn = args[0]
        v = args[1]
        jvp_input = args[2:]
        output = fn(*jvp_input)

        gradient_output = self.compute_jvp(fn, v, jvp_input, output)
        return output, gradient_output


class _LinearizeInner(_JvpInner):
    """
    Compute the jacobian-vector-product of the given network. This Class is mainly useful
    if you want to apply jvp multiple times under the same input and fn.
    """
    def construct(self, *args):
        fn = args[0]
        v = args[1]
        output = args[2]
        jvp_input = args[3]
        gradient_output = self.compute_jvp(fn, v, jvp_input, output)
        return gradient_output


class Vjp(Cell):
    """
    Vjp will be deprecated in the future, please use :func:`mindspore.ops.vjp` instead.

    Supported Platforms:
        Deprecated
    """

    @deprecated("1.9", "ops.vjp", True)
    def __init__(self, fn):
        super(Vjp, self).__init__()
        self.fn = fn
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.grad_single_value = C.GradOperation(sens_param=True)
        self.issubclass_ = inner.IsSubClass()
        self.typeof = Primitive('typeof')

    @jit
    def construct(self, *args):
        front_input = args[0:-1]
        output = self.fn(*front_input)
        if len(front_input) == 1:
            gradient_output = self.grad_single_value(self.fn)(*args)
        else:
            gradient_output = self.grad(self.fn)(*args)
        return output, gradient_output
