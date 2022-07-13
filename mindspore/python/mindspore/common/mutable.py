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
"""mutable function for setting constants mutable."""

from ..common.tensor import Tensor


class _Tuple(tuple):
    pass


class _List(list):
    pass


class _Dict(dict):
    pass


def _check_all_tensor(value):
    """Check if all the elements are Tensor."""
    if isinstance(value, (tuple, list)):
        for element in value:
            if not _check_all_tensor(element):
                return False
        return True
    if isinstance(value, dict):
        for element in value.values():
            if not _check_all_tensor(element):
                return False
        return True
    return isinstance(value, Tensor)


def mutable(input_data):
    """
    Make a constant value mutable.

    Currently, all the inputs of Cell except Tensor such as scalar, tuple, list and dict, are regarded as constant
    values. The constant values are non-differentiable and used to do constant folding in the optimization process.

    Besides, currently when the network input is tuple[Tensor], list[Tensor] or dict[Tensor], even without changing
    the shape and dtype of the Tensors, the network will be re-compiled when calling this network repeatedly because
    the these inputs are regarded as constant values.

    To solve the above problems, we provide api `mutable` to make the constant inputs of Cell 'mutable'. A 'mutable'
    input means that it is changed to be a variable input just like Tensor and the most important thing is that it
    will be differentiable.

    Args:
        input_data (Union[Tensor, tuple[Tensor], list[Tensor], dict[Tensor]]): The input data to be made mutable.

    .. warning::
        - This is an experimental prototype that is subject to change or deletion.
        - The runtime has not yet supported to handle the scalar data flow. So we only support tuple[Tensor],
          list[Tensor] or dict[Tensor] for network input to avoid the re-compiled problem now.
        - Tensor is mutable by default, when the `input_data` is Tensor, we just return the origin Tensor and nothing
          is done.
        - Currently we only support to use this api outside the network temporarily.
        - Currently this api only works in GRAPH mode.

    Returns:
        The origin input data which has been set mutable.

    Raises:
        TypeError: If `input_data` is not one of Tensor, tuple[Tensor], list[Tensor], dict[Tensor] or their nested
            structure.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore.ops.composite import GradOperation
        >>> from mindspore.common import mutable
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.matmul = ops.MatMul()
        ...
        ...     def construct(self, z):
        ...         x = z[0]
        ...         y = z[1]
        ...         out = self.matmul(x, y)
        ...         return out
        ...
        >>> class GradNetWrtX(nn.Cell):
        ...     def __init__(self, net):
        ...         super(GradNetWrtX, self).__init__()
        ...         self.net = net
        ...         self.grad_op = GradOperation()
        ...
        ...     def construct(self, z):
        ...         gradient_function = self.grad_op(self.net)
        ...         return gradient_function(z)
        ...
        >>> z = mutable((Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
        ...              Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)))
        >>> output = GradNetWrtX(Net())(z)
        >>> print(output)
        (Tensor(shape=[2, 3], dtype=Float32, value=
        [[ 1.41000009e+00, 1.60000002e+00, 6.59999943e+00],
         [ 1.41000009e+00, 1.60000002e+00, 6.59999943e+00]]), Tensor(shape=[3, 3], dtype=Float32, value=
        [[ 1.70000005e+00, 1.70000005e+00, 1.70000005e+00],
         [ 1.89999998e+00, 1.89999998e+00, 1.89999998e+00],
         [ 1.50000000e+00, 1.50000000e+00, 1.50000000e+00]]))
    """

    if isinstance(input_data, Tensor):
        return input_data

    if not _check_all_tensor(input_data):
        raise TypeError(
            f"For 'mutable', the 'input_data' should be one of (Tensor, tuple[Tensor], list[Tensor], dict[Tensor]) "
            f"or their nested structures, but got {input_data}.")

    ret = input_data
    if isinstance(input_data, list):
        ret = _List(input_data)
    elif isinstance(input_data, tuple):
        ret = _Tuple(input_data)
    elif isinstance(input_data, dict):
        ret = _Dict(input_data)

    setattr(ret, "__ms_mutable__", True)
    return ret
