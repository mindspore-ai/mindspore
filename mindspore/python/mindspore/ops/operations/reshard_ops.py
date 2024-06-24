# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Operators for reshard."""
from mindspore.ops.primitive import Primitive, prim_attr_register


class Reshard(Primitive):
    r"""
    Reshard the tensor by the given in_layout and out_layout, which can precisely
    define how the dimension of the tensor and the device clusters be sharded in
    parallel procedure.

    Note:
        - The in and out layout should be the type mindspore.Layout.
        - The in and out layout should be the same value of layout when invoke
          ops.Reshard(layout, layout, in_strategy).
        - The in_strategy should be the strategy derived from the layout.
        - This primitive is not recommended to use directly. We recommend to use mindspore.reshard.

    Inputs:
        - **tensor** (Tensor) - The tensor to be resharded.

    Outputs:
        Tensor. The mathematically equivalent of the input tensor.

    Examples:
        >>> from mindspore.parallel.shard import Layout
        >>> _layout = Layout((4, 2), ("dp", "mp"))
        >>> layout = (_layout("dp", "mp"),)
        >>> reshard = ops.Reshard(layout, layout, in_strategy)
        >>> reshard(tensor)
    """

    @prim_attr_register
    def __init__(self, in_layout, out_layout, in_strategy):
        super().__init__(name="Reshard")
        self.shard(in_layout, out_layout)
        self.in_strategy = in_strategy

    def __call__(self, tensor):
        return tensor
