# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Symbol implementation."""

__all__ = ['Symbol']


class Symbol:
    r"""
    Symbol is a data structure to indicate the symbolic info of shape.

    Args:
        max (int): The maximum length of this dimension, which is valid when it's greater than 'min' value. Default: 0.
        min (int): The minimum length of this dimension. Default: 1.
        divisor (int): The divisor(:math:`d`) when symbol is represented by :math:`d * N + r, N \ge 1`. Default: 1.
        remainder (int): The remainder(:math:`r`) when symbol is represented by :math:`d * N + r, N \ge 1`. Default: 0.
        unique (bool): When the symbol object is used multiple times, if 'unique' is True, the shape items of this
            symbol are considered to be same length, otherwise only symbol info is shared by multiple dimensions.

    Outputs:
        Symbol.

    Raises:
        ValueError: If 'min' is not positive value.
        ValueError: If 'divisor' is not positive value.
        ValueError: If 'remainder' is not in the range "[0, divisor)".
    """

    def __init__(self, max=0, min=1, divisor=1, remainder=0, unique=False, **kawgs):
        if min <= 0:
            raise ValueError("For 'Symbol', the 'min' value should be positive, but got {}".format(min))
        if divisor <= 0:
            raise ValueError("For 'Symbol', the 'divisor' value should be positive, but got {}".format(divisor))
        if remainder < 0 or remainder >= divisor:
            raise ValueError(
                "For 'Symbol', the 'remainder' value should be in the range '[0, {})', but got {}".format(
                    divisor, remainder))
        self.max = max
        self.min = min
        self.divisor = divisor
        self.remainder = remainder
        self.unique = unique
        self.id = id(self)

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Convert the symbolic info to dictionary."""
        res = {}
        if self.max > self.min:
            res["max"] = self.max
        if self.min > 1:
            res["min"] = self.min
        if self.divisor != 1:
            res["divisor"] = self.divisor
        if self.remainder != 0:
            res["remainder"] = self.remainder
        if self.unique:
            res["id"] = self.id
        return res
