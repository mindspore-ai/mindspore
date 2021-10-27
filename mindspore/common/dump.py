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
"""Controlling dump behavior."""

from mindspore._c_expression import security


def set_dump(target, enabled=True):
    """
    Enable or disable dump for the cell instance and its contents.

    The default enabled status for a cell is False. Please note that this
    mode takes effect only when the dump_mode field in dump config file is
    2. See the `dump document <https://mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html>`_
    for details.

    .. warning::
        This is an experimental prototype that is subject to change and/or
        deletion.

    Note:
        1. This API is only effective for GRAPH_MODE with Ascend backend.
        2. When input is a cell, this API is only effective for the members of
            the cell instance. If an operator is not a member of the cell
            instance, the dump flag will not be set for this operator (e.g.
            functional operators used directly in construct method). To make
            this API effective, please use self.some_op = SomeOp() in cell
            __init__ method.

    Args:
        target (Union[Cell, Primitive]): The Cell instance or Primitive instance
            to which the dump flag is set.
        enabled (bool): True means enable dump, False means disable dump.
            Default: True.

    Examples:
        >>> from mindspore.nn import Cell
        >>> class MyNet(Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = nn.Conv2d(5, 6, 5, pad_mode='valid')
        ...         self.relu1 = nn.ReLU()
        ...
        ...     def construct(self, x):
        ...         x = self.conv1(x)
        ...         x = self.relu1(x)
        ...         return x
        >>> net = MyNet()
        >>> set_dump(net.conv1)
    """
    if security.enable_security():
        raise ValueError('The set_dump API is not supported, please recompile '
                         'source without "-s on".')

    import mindspore.nn as nn  # avoid circular import
    from mindspore.ops import Primitive
    if not isinstance(target, nn.Cell) and not isinstance(target, Primitive):
        raise ValueError(f"The \"target\" parameter must be an instance of "
                         f"Cell or Primitive, "
                         f"but got an instance of {type(target)}.")

    if not isinstance(enabled, bool):
        raise ValueError("The \"enabled\" parameter must be bool.")

    mode = "true" if enabled else "false"
    if isinstance(target, nn.Cell):
        primitives = getattr(target, "_primitives", {})
        for value in primitives.values():
            if value:
                value.add_prim_attr("dump", mode)
        for cell in target.cells():
            set_dump(cell, enabled)
        return

    if isinstance(target, Primitive):
        target.add_prim_attr("dump", mode)
        return
