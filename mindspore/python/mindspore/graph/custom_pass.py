# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""
Experimental custom pass registration functionality.

This module provides experimental APIs for registering custom optimization passes.
These APIs are subject to change and should be used with caution in production code.
"""

__all__ = ["CustomPassType", "register_custom_pass"]

from mindspore import _c_expression


class CustomPassType:
    """
    Custom pass type.
    Possible enumeration values include: ``CustomPassType.PATTERN`` , ``CustomPassType.FULL_GRAPH`` .

    - ``CustomPassType.PATTERN``: Rewrites the graph structure based on a user-defined pattern to match,
      such as replacing an `Add` operator and a `Neg` operator with a `Sub` operator.
      This is applied in the backend optimization process.

    - ``CustomPassType.FULL_GRAPH``: Rewrites the graph based on the information of the entire graph.
      This is applied in the frontend optimization process.
    """
    PATTERN = _c_expression.CustomPassType.PATTERN
    FULL_GRAPH = _c_expression.CustomPassType.FULL_GRAPH


def register_custom_pass(
    pass_name: str,
    plugin_so_path: str,
    device: str = "all",
    pass_type: CustomPassType = CustomPassType.PATTERN
) -> bool:
    """Register a custom pass to modify graph structure for default ``"ms_backend"`` backend.

    .. note::
        Windows and MacOS systems are not supported yet.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        pass_name (str): Name of the pass expected to be provided by the plugin.
        plugin_so_path (str): Absolute path to the plugin shared library (.so file).
        device (str, optional): Target device for the pass. Supported values: ``"cpu"`` , ``"gpu"`` , ``"ascend"`` ,
            or ``"all"`` . Default: ``"all"`` .
        pass_type (CustomPassType, optional): Type of the custom pass.
            Default: ``CustomPassType.PATTERN``.
            It can be ``CustomPassType.PATTERN``, ``CustomPassType.FULL_GRAPH``.

            - ``CustomPassType.PATTERN``: rewrites the graph structure based on a user-defined pattern to match,
              such as replacing an `Add` operator and a `Neg` operator with a `Sub` operator.
              This is applied in the backend optimization process.
            - ``CustomPassType.FULL_GRAPH``: rewrites the graph based on the information of the entire graph.
              This is applied in the frontend optimization process.


    Returns:
        bool. Returns ``True`` if the custom pass is registered successfully, otherwise returns ``False``.

    Raises:
        TypeError: If `pass_type` is not of type :class:`mindspore.graph.CustomPassType` .

    Examples:
        >>> import mindspore.graph as graph
        >>> # Register a custom optimization pass
        >>> success = graph.register_custom_pass(
        ...     pass_name="my_fusion_pass",
        ...     plugin_so_path="/path/to/my_plugin.so",
        ...     device="ascend"
        ... )
        >>> print(f"Registration successful: {success}")
    """
    if not isinstance(pass_type, _c_expression.CustomPassType):
        raise TypeError(
            f"pass_type must be of type CustomPassType, got {type(pass_type).__name__}")
    return _c_expression.register_custom_pass(pass_name, plugin_so_path, device, pass_type)
