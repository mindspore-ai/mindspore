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
    An enum for quant datatype, contains `INT2`~`INT8`, `UINT2`~`UINT8`.
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

    def __str__(self):
        return f"{self.name}"

    @staticmethod
    def is_signed(dtype):
        """
        Get whether the quant datatype is signed.

        Args:
            dtype (QuantDtype): quant datatype.

        Returns:
            bool, whether the input quant datatype is signed.

        Examples:
            >>> quant_dtype = QuantDtype.INT8
            >>> is_signed = QuantDtype.is_signed(quant_dtype)
        """
        return dtype in [QuantDtype.INT2, QuantDtype.INT3, QuantDtype.INT4, QuantDtype.INT5,
                         QuantDtype.INT6, QuantDtype.INT7, QuantDtype.INT8]

    @staticmethod
    def switch_signed(dtype):
        """
        Switch the signed state of the input quant datatype.

        Args:
            dtype (QuantDtype): quant datatype.

        Returns:
            QuantDtype, quant datatype with opposite signed state as the input.

        Examples:
            >>> quant_dtype = QuantDtype.INT8
            >>> quant_dtype = QuantDtype.switch_signed(quant_dtype)
        """
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
    def _value(self):
        """The value of the Enum member."""
        return int(re.search(r"(\d+)", self._value_).group(1))

    @DynamicClassAttribute
    def num_bits(self):
        """
        Get the num bits of the QuantDtype member.

        Returns:
            int, the num bits of the QuantDtype member

        Examples:
            >>> quant_dtype = QuantDtype.INT8
            >>> num_bits = quant_dtype.num_bits
        """
        return self._value
