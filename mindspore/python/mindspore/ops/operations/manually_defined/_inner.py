# Copyright 2023 Huawei Technologies Co., Ltd
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
Inner-defined operators.
"""

import numbers
import numpy as np
from mindspore.common import dtype as mstype
from mindspore import _checkparam as validator
from mindspore.common._decorator import deprecated
from mindspore.ops.primitive import prim_attr_register, Primitive


class ScalarCast(Primitive):
    """
    'ops.ScalarCast' is deprecated from version 2.3 and will be removed in a future version,
    please use `int(x)` or `float(x)` instead.

    Supported Platforms:
        Deprecated

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> scalar_cast = ops.ScalarCast()
        >>> output = scalar_cast(255.0, mindspore.int64)
        >>> print(output)
        255
    """

    @deprecated("2.3", "ops.ScalarCast", False)
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'input_y'], outputs=['output_data'])

    def __call__(self, input_x, input_y):
        validator.check_value_type("x", input_x, [bool, numbers.Number], self.name)
        if input_y not in (mstype.int64, mstype.float64, mstype.bool_):
            raise ValueError(f"For 'ScalarCast', the supported type is in the list: "
                             f"[mindspore.int64, mindspore.float64, mindspore.bool], but got {input_y}")
        dtype = input_y
        if isinstance(dtype, type(mstype.tensor_type)):
            dtype = dtype.element_type()
        np_dtype = str(dtype)
        value = np.cast[np_dtype.lower()](input_x)
        value = value.item()
        return value
