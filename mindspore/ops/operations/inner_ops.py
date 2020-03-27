# Copyright 2020 Huawei Technologies Co., Ltd
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

"""inner_ops"""

from ...common.dtype import tensor, dtype_to_pytype
from ..primitive import prim_attr_register, PrimitiveWithInfer


class ScalarCast(PrimitiveWithInfer):
    """
    Cast the input scalar to another type.

    Inputs:
        - **input_x** (scalar) - The input scalar. Only constant value is allowed.
        - **input_y** (mindspore.dtype) - The type should cast to be. Only constant value is allowed.

    Outputs:
        Scalar. The type is same as the python type corresponding to `input_y`.

    Examples:
        >>> scalar_cast = P.ScalarCast()
        >>> output = scalar_cast(255.0, mindspore.int32)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __infer__(self, x, t):
        value, to = x['value'], t['value']
        if value is not None:
            if isinstance(to, type(tensor)):
                to = to.element_type()
            np_type = dtype_to_pytype(to)
            value = np_type(value)
        out = {'shape': x['shape'],
               'dtype': t['value'],
               'value': value}
        return out
