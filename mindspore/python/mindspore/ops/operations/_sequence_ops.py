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
"""Operations for sequence"""

from mindspore.ops.primitive import Primitive, prim_attr_register


class ListAppend(Primitive):
    r"""
    Append element to the end of list.

    .. note::
        This operation is used for dynamic length list and this it is only for internal used.

    Inputs:
        - **input_data** (List) - The list for target to append. Must be dynamic length sequence
        - **target** (Any Object) - The target element to be appended. The shape and type of target must be the same as
          as the element within 'input_data'.

    Outputs:
        Dynamic length list after append.

    Raises:
        TypeError: The 'input_data' is not dynamic length list.
        ValueError: The shape or type of 'target' is not the same as the element within 'input_data'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ListAppend"""
        self.init_prim_io_names(inputs=['input_data', 'target'], outputs=['output_data'])
