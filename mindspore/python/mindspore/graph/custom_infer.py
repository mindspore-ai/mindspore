# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Custom Infer for operators."""

from mindspore._c_expression import OpInferRegistry
from mindspore._c_expression import OpWrapperRegistry


def register_op_infer(op_name, infer_name):
    """
    Register a custom inference function for an operator.
    
    This decorator allows users to register custom inference functions for operators in MindSpore.
    The registered inference function will be called during graph compilation to compute custom
    attributes for the operator's output. The result of the inference function will be stored
    in the output of the operator, making it accessible to subsequent operations in its 
    function registered in `register_op_wrapper` or `register_op_infer`.

    Note:
        - Multiple inference functions can be registered for the same operator using
          different `infer_name` values.
        - The inference function will receive the same arguments as the operator it
          is registered for.
        - The inference result will be propagated through the computation graph and
          can be accessed by subsequent operations.
        - This registration affects all instances of the specified operator in the
          current process.
    
    Args:
        op_name (str): The name of the operator to register the inference function for.
            Must be a non-empty string.
        infer_name (str): The name of the custom attr. This name will be used as
            the attr name of the output of the operator. Must be a non-empty string.
    
    Returns:
        callable. A decorated function that is registered as custom inference function
    
    Raises:
        ValueError: If `op_name` or `infer_name` is not a non-empty string.
        RuntimeError: If an inference function is registered for the same `infer_name`
            of the same operator multiple times.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops, jit
        >>> from mindspore.graph import register_op_infer
        >>> # Register a custom layout inference for the Add operator, the custom layout can
        >>> # be obtained during inferencing the subsequent operator Mul:
        >>> class MyNet(nn.Cell):
        ...     @jit
        ...     def construct(self, x, y):
        ...         z = ops.Add()(x, y)
        ...         result = ops.Mul()(z, x)
        ...         return result
        >>>
        >>> @register_op_infer("Add", 'layout')
        >>> def infer_add_layout(x, y):
        ...     return x.shape + y.shape
        >>>
        >>> @register_op_infer("Mul", 'layout')
        >>> def infer_mul_layout(x, y):
        ...     print(x.layout)
        >>>
        >>> net = MyNet()
        >>> input_x = Tensor([[1., 2., 3.], [4., 5., 6.]])
        >>> input_y = Tensor([[1., 2., 3.], [4., 5., 6.]])
        >>> net(input_x, input_y)
        (2, 3, 2, 3)
    """
    if not isinstance(op_name, str) or not op_name.strip():
        raise ValueError("op_name must be a non-empty string")
    if not isinstance(infer_name, str) or not infer_name.strip():
        raise ValueError("infer_name must be a non-empty string")

    def decorator(func):
        OpInferRegistry.instance().register_function(op_name, infer_name, func)
        return func
    return decorator


def clear_op_infer():  # pylint: disable=missing-docstring
    OpInferRegistry.instance().clear_funcs()


def register_op_wrapper(op_name):
    """
    Register a custom wrapper function to replace the node of an operator in computation graph.
    
    This decorator allows users to register custom wrapper functions that replace the computation 
    logic of operators in computation graph. When the JIT compiler compiles the specified operator,
    it will compile the registered wrapper function instead of the original operator.

    The wrapper function receives the same arguments as the original operator and must return
    a callable function.

    Note:
        - Only one wrapper function can be registered per operator. Attempting to register
          multiple wrappers for the same operator will raise an exception.
        - The replacement affects all instances of the specified operator within JIT-compiled
          Cells in the current process.
        - The wrapper function arguments should match the original operator's signature.

    Args:
        op_name (str): The name of the operator to register the wrapper function for.
            Must be a non-empty string.
    
    Returns:
        callable. A decorated function that is registered as op wrapper function
    
    Raises:
        ValueError: If `op_name` is not a non-empty string.
        RuntimeError: If attempting to register a wrapper for the same `op_name` multiple times.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    
    Examples:
        >>> from mindspore import Tensor, ops, jit
        >>> from mindspore.graph import register_op_wrapper
        >>> # Replace Add operation with custom logic (x + 2*y):
        >>> @register_op_wrapper("Add")
        >>> def double_add_wrapper(x, y):
        ...     def func(x, y):
        ...         return x + 2 * y
        ...     return func
        >>> 
        >>> class MyNet(nn.Cell):
        ...     @jit
        ...     def construct(self, x, y):
        ...         return ops.Add()(x, y)  # Will execute x + 2*y
    """
    if not isinstance(op_name, str) or not op_name.strip():
        raise ValueError("op_name must be a non-empty string")

    def decorator(func):
        OpWrapperRegistry.instance().register_function(op_name, func)
        return func
    return decorator


def clear_op_wrapper():  # pylint: disable=missing-docstring
    OpWrapperRegistry.instance().clear_funcs()
