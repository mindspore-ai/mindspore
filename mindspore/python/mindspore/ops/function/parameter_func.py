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

"""Defines parameter operators with functional form."""

from mindspore.ops import operations as P


assign_ = P.Assign()
def assign(variable, value):
    """
    Assigns `Parameter` with a value.

    Args of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        variable (Parameter): The `Parameter`. :math:`(N,*)` where :math:`*` means,
            any number of additional dimensions, its rank should be less than 8.
        value (Tensor): The value to be assigned, has the same shape with `variable`.

    Returns:
        Tensor, has the same data type and shape as original `variable`.

    Raises:
        TypeError: If `variable` is not a Parameter.
        TypeError: If `value` is not a Tensor.
        RuntimeError: If the data type of `variable` and `value` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> value = Tensor([2.0], mindspore.float32)
        >>> variable = mindspore.Parameter(Tensor([1.0], mindspore.float32), name="variable")
        >>> output = ops.assign(variable, value)
        >>> print(output)
        [2.]
    """
    return assign_(variable, value)


assign_sub_ = P.AssignSub()
def assign_sub(variable, value):
    """
    Updates a `Parameter` by subtracting a value from it.

    Args of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.
    If `value` is a number, the number is automatically converted to Tensor,
    and the data type is consistent with the Tensor data type involved in the operation.

    Note:
        Since `variable` is a data type Parameter, the data type cannot be changed,
        so only the type of `value` is allowed to be promoted to the type of `variable`.
        And the conversion type supported by different devices will be different,
        it is recommended to use the same data type when using this operator.

    Args:
        variable (Parameter): The `Parameter`.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank be should be less than 8.
        value (Union[numbers.Number, Tensor]): The value to be subtracted from the `variable`.
            It must have the same shape as `variable` if it is a Tensor.
            it is recommended to use the same data type when using this operator.

    Returns:
        Tensor, has the same data type and shape as original `variable`.

    Raises:
        TypeError: If `value` is neither Number nor Tensor.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> variable = mindspore.Parameter(initializer(1, [1], mindspore.int32), name="global_step")
        >>> value = Tensor(np.ones([1]).astype(np.int32) * 100)
        >>> output = ops.assign_sub(variable, value)
        >>> print(output)
        [-99]
    """
    return assign_sub_(variable, value)


assign_add_ = P.AssignAdd()
def assign_add(variable, value):
    """
    Updates a `Parameter` by adding a value to it.

    Args of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.
    If `value` is a number, the number is automatically converted to Tensor,
    and the data type is consistent with the Tensor data type involved in the operation.

    Note:
        Since `variable` is a data type Parameter, the data type cannot be changed,
        so only the type of `value` is allowed to be promoted to the type of `variable`.
        And the conversion type supported by different devices will be different,
        it is recommended to use the same data type when using this operator.

    Args:
        variable (Parameter): The `Parameter`.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        value (Union[numbers.Number, Tensor]): The value to be added to the `variable`.
            It must have the same shape as `variable` if it is a Tensor.
            it is recommended to use the same data type when using this operator.

    Returns:
        Tensor, has the same data type and shape as original `variable`.

    Raises:
        TypeError: If `value` is neither Number nor Tensor.
        RuntimeError: If the data type of `variable` and `value` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> variable = mindspore.Parameter(initializer(1, [1], mindspore.int32), name="global_step")
        >>> value = Tensor(np.ones([1]).astype(np.int32) * 100)
        >>> output = ops.assign_add(variable, value)
        >>> print(variable.asnumpy())
        [101]
    """
    return assign_add_(variable, value)


__all__ = [
    'assign',
    'assign_sub',
    'assign_add'
]
__all__.sort()
