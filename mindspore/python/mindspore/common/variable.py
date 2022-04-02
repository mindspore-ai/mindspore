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
"""Variable class for setting constants mutable."""

from .._c_expression import Variable_
from ..common.tensor import Tensor, CSRTensor, COOTensor


class Variable(Variable_):
    """
    Currently, all the inputs of Cell except Tensor such as scalar, tuple, list and dict, are regarded as constant
    values. The constant values are non-differentiable and used to do constant folding in the optimization process.
    We provide a class 'Variable' to store a constant value, to make the constant inputs of Cell 'mutable'.
    A 'mutable' constant input means that it is changed to be a variable input just like Tensor and the most important
    thing is that it is differentiable from now on.

    .. warning::
        This is an experimental prototype that is subject to change or deletion.

    Args:
        value (Union[bool, float, int, tuple, list, dict, Tensor]): The value to be stored.

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore.ops.composite import GradOperation
        >>> from mindspore.common.variable import Variable
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...
        ...     def construct(self, x, y):
        ...         return x * y
        ...
        >>> class GradNet(nn.Cell):
        ...     def __init__(self, net):
        ...         super(GradNet, self).__init__()
        ...         self.net = net
        ...         self.grad_op = GradOperation()
        ...
        ...     def construct(self, x, y):
        ...         gradient_function = self.grad_op(self.net)
        ...         return gradient_function(x, y)
        ...
        >>> x = Variable(2)
        >>> output = GradNet(Net())(x, 3)
        >>> print(output)
        3
    """

    def __init__(self, value):
        if not isinstance(value, (bool, int, float, tuple, list, dict, Tensor, COOTensor, CSRTensor)):
            raise TypeError(
                f"For 'Varibale', the 'value' should be one of (int, float, tuple, list, dict, Tensor, COOTensor, "
                f"CSRTensor), but got {type(value).__name__}")
        Variable_.__init__(self, value)
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
