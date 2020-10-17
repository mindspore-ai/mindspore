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
"""Constant module for compression"""
import enum
import re
from types import DynamicClassAttribute


__all__ = ["QuantDtype"]


@enum.unique
class QuantDtype(enum.Enum):
    """
    For type switch
    """
    INT2 = "INT2"
    INT3 = "INT3"
    INT4 = "INT4"
    INT5 = "INT5"
    INT6 = "INT6"
    INT7 = "INT7"
    INT8 = "INT8"

    UINT2 = "UINT2"
    UINT3 = "UINT3"
    UINT4 = "UINT4"
    UINT5 = "UINT5"
    UINT6 = "UINT6"
    UINT7 = "UINT7"
    UINT8 = "UINT8"

    FLOAT16 = "FLOAT16"
    FLOAT32 = "FLOAT32"

    def __str__(self):
        return f"{self.name}"

    @staticmethod
    def is_signed(dtype):
        return dtype in [QuantDtype.INT2, QuantDtype.INT3, QuantDtype.INT4, QuantDtype.INT5,
                         QuantDtype.INT6, QuantDtype.INT7, QuantDtype.INT8]

    @staticmethod
    def switch_signed(dtype):
        """switch signed"""
        type_map = {
            QuantDtype.INT2: QuantDtype.UINT2,
            QuantDtype.INT3: QuantDtype.UINT3,
            QuantDtype.INT4: QuantDtype.UINT4,
            QuantDtype.INT5: QuantDtype.UINT5,
            QuantDtype.INT6: QuantDtype.UINT6,
            QuantDtype.INT7: QuantDtype.UINT7,
            QuantDtype.INT8: QuantDtype.UINT8,
            QuantDtype.UINT2: QuantDtype.INT2,
            QuantDtype.UINT3: QuantDtype.INT3,
            QuantDtype.UINT4: QuantDtype.INT4,
            QuantDtype.UINT5: QuantDtype.INT5,
            QuantDtype.UINT6: QuantDtype.INT6,
            QuantDtype.UINT7: QuantDtype.INT7,
            QuantDtype.UINT8: QuantDtype.INT8
        }
        return type_map[dtype]

    @DynamicClassAttribute
    def value(self):
        """The value of the Enum member."""
        return int(re.search(r"(\d+)", self._value_).group(1))

    @DynamicClassAttribute
    def num_bits(self):
        """The num_bits of the Enum member."""
        return self.value
