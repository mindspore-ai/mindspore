# Copyright 2025 Huawei Technologies Co., Ltd
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

"""custom backend registration functionality."""

import os
from mindspore import _c_expression


def register_custom_backend(backend_name: str, backend_path: str) -> bool:
    """Register a custom backend with MindSpore and use it for model compilation and execution.

    .. note::
        - Custom backend only takes effect when using @jit, and the backend parameter
          in the @jit decorator must be consistent with the registered backend name.
        - This interface only supports Linux systems.

    Args:
        backend_name (str): Name of the backend expected to be provided by the plugin.
        backend_path (str): Absolute path to the plugin shared library (.so file).

    Returns:
        bool. Return True if the custom backend is successfully registered; otherwise, False is returned.

    Raises:
        ValueError: If the custom backend path does not exist or the file is invalid.

    Examples:
        >>> import mindspore.graph as graph
        >>> from mindspore import mint, jit
        >>> # Register a custom backend
        >>> success = graph.register_custom_backend(
        ...     backend_name="my_backend",
        ...     backend_path="/path/to/my_backend.so",
        ... )
        >>> print(f"Registration successful: {success}")
        >>> # Use the custom backend
        >>> @jit(backend="my_backend")
        ...     def my_func(x):
        ...         return mint.sin(x)
    """
    if backend_name is None or backend_path is None:
        raise ValueError("Both backend_name and backend_path must be provided.")
    if not os.path.isfile(backend_path) or not os.access(backend_path, os.R_OK):
        raise ValueError("The file {} does not exist or permission denied!".format(backend_path))
    if not backend_path.endswith(".so"):
        raise ValueError("The file {} is not a shared library!".format(backend_path))
    return _c_expression.register_custom_backend(backend_name, backend_path)


__all__ = ["register_custom_backend"]
