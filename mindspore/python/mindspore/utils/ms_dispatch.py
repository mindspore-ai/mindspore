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
"""MsDispatchMode - Context manager for operator dispatch interception."""

__all__ = ["MsDispatchMode"]

import contextlib
from mindspore._c_expression import _DispatchGuardStack


class MsDispatchMode(contextlib.ContextDecorator):
    """
    A context manager for intercepting operator dispatch.

    When entering the context, all operators executed within the scope will first invoke
    the ``__ms_dispatch__`` callback method. This allows users to customize operator
    behavior, such as logging, profiling, or modifying inputs/outputs.

    MsDispatchMode supports nesting. When nested, the innermost mode takes precedence.
    After exiting a nested context, the outer mode resumes handling operator dispatch.

    To use this class, inherit from it and override the ``__ms_dispatch__`` method.

    Args:
        None

    Note:
        - This context manager can also be used as a decorator.
        - The ``__ms_dispatch__`` method receives an ``OpFunc`` object that wraps
          the original operator call, along with the operator's arguments.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>>
        >>> class OpLogger(ms.MsDispatchMode):
        ...     '''Log all operator calls.'''
        ...     def __ms_dispatch__(self, func, args, kwargs=None):
        ...         print(f"Executing op: {func.name}")
        ...         result = func(*args)
        ...         print(f"Op {func.name} completed")
        ...         return result
        ...
        >>> x = Tensor([1.0, 2.0, 3.0])
        >>> y = Tensor([4.0, 5.0, 6.0])
        >>>
        >>> # Single context usage
        >>> with OpLogger():
        ...     z = x + y  # Will print log messages
        Executing op: Add
        Op Add completed
        >>>
        >>> # Nested context usage
        >>> class InnerGuard(ms.MsDispatchMode):
        ...     def __ms_dispatch__(self, func, args, kwargs=None):
        ...         print(f"[Inner] {func.name}")
        ...         return func(*args)
        ...
        >>> with OpLogger():
        ...     a = x * 2  # OpLogger handles this
        ...     with InnerGuard():
        ...         b = x + y  # InnerGuard handles this (inner takes precedence)
        ...     c = x - y  # OpLogger handles this again
        Executing op: Mul
        Op Mul completed
        [Inner] Add
        Executing op: Sub
        Op Sub completed
        >>>
        >>> # Decorator usage
        >>> @OpLogger()
        ... def my_function(a, b):
        ...     return a + b
        ...
        >>> result = my_function(x, y)  # Will log the Add operation
        Executing op: Add
        Op Add completed

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __enter__(self):
        """Enter the dispatch guard context."""
        _DispatchGuardStack.push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the dispatch guard context."""
        _DispatchGuardStack.pop()
        return False

    def __ms_dispatch__(self, func, args, kwargs=None):
        """
        Callback method invoked for each operator within this context.

        Override this method in subclasses to customize operator dispatch behavior.

        Args:
            func (OpFunc): A callable wrapper for the original operator.
                - ``func.name``: The name of the operator (e.g., "Add", "MatMul").
                - ``func(*args)``: Invokes the original operator with given arguments.
            args (tuple): Positional arguments passed to the operator.
            kwargs (dict, optional): Keyword arguments passed to the operator. Default: ``None``.

        Returns:
            The result of the operator execution. By default, this method calls the
            original operator and returns its result unchanged.

        Note:
            - To execute the original operator, call ``func(*args)`` within this method.
            - You can modify the arguments before calling, or modify the result after.
            - If you don't call ``func(*args)``, the original operator won't execute.

        Examples:
            >>> class MyGuard(ms.MsDispatchMode):
            ...     def __ms_dispatch__(self, func, args, kwargs=None):
            ...         # Log before execution
            ...         print(f"Before: {func.name}")
            ...         # Execute the original operator
            ...         result = func(*args)
            ...         # Log after execution
            ...         print(f"After: {func.name}")
            ...         return result
        """
        # Default implementation: just call the original operator
        return func(*args)
