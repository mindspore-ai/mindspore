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
"""Operations for sequence"""

from mindspore.ops.primitive import Primitive, prim_attr_register
from .manually_defined import ScalarAdd, ScalarBool, ScalarDiv, ScalarMul, ScalarEq, ScalarFloorDiv, ScalarGe, \
    ScalarGt, ScalarLe, ScalarLog, ScalarLt, ScalarMod, ScalarPow, ScalarSub, ScalarUadd, ScalarUsub


class bool_not(Primitive):
    r"""
    Returns bool_not `not` of bool input.

    .. note::
        The inputs can be constant/variable value. Usage is the same as 'not' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar, the type can be bool.

    Outputs:
        Scalar, the type is bool.

    Raises:
        TypeError: If `x` are not bool scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize bool_not"""

    def __call__(self, x):
        return not x


class bit_and(Primitive):
    r"""
    Returns bitwise `and` of two scalars.

    .. math::

        out_{i} = x_{i} \text{ % } y_{i}

    .. note::
        The inputs can be constant/variable value. Usage is the same as '%' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar, the type can be int or bool.
        - **y** (Scalar) - A constant or variable scalar, the type can be int or bool.

    Outputs:
        Scalar, the type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScalarMod"""


class bit_or(Primitive):
    r"""
    Returns bitwise `or` of two scalars.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '|' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar, the type can be int or bool.
        - **y** (Scalar) - A constant or variable scalar, the type can be int or bool.

    Outputs:
        Scalar, the type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScalarMod"""
