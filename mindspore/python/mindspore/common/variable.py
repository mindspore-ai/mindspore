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
from ..common.tensor import Tensor


class Variable(Variable_):
    """
    Currently, all the inputs of Cell except Tensor such as scalar, tuple, list and dict, are regarded as constant
    values. The constant values are non-differentiable and used to do constant folding in the optimization process.
    We provide a class 'Variable' to store a constant value, to make the constant inputs of Cell 'mutable'.
    A 'mutable' constant input means that it is changed to be a variable input just like Tensor and the most important
    thing is that it is differentiable from now on.

    Besides, currently when the network input is tuple[Tensor], list[Tensor] or dict[Tensor], if the value of tensor is
    changed without changing the shape and dtype, the network will be re-compiled because the these inputs are regarded
    as constant values. Now we can avoid this problem by using 'Variable' to store these inputs.

    .. warning::
        - This is an experimental prototype that is subject to change or deletion.
        - The runtime has not yet supported to handle the scalar data flow. So we only support tuple[Tensor],
          list[Tensor] or dict[Tensor] for network input to avoid the re-compiled problem now.

    Args:
        value (Union[tuple[Tensor], list[Tensor], dict[Tensor]]): The value to be stored.

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
        if isinstance(value, Tensor) or not self._check_all_tensor(value):
            raise TypeError(
                f"For 'Varibale', the 'value' should be one of (tuple[Tensor], list[Tensor], dict[Tensor]) "
                f"or their nested structures, but got {value}")
        Variable_.__init__(self, value)
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def _check_all_tensor(self, value):
        """Check if all the elements are Tensor."""
        if isinstance(value, (tuple, list)):
            for element in value:
                if not self._check_all_tensor(element):
                    return False
            return True
        if isinstance(value, dict):
            for element in value.values():
                if not self._check_all_tensor(element):
                    return False
            return True
        return isinstance(value, Tensor)
