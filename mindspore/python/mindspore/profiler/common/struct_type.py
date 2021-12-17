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
"""Define a cpp type map python struct type."""

from enum import Enum


class StructType(Enum):
    """Define cpp struct type, value is python struct type."""
    CHAR = 'c'
    UCHAR = 'B'
    UINT16 = 'H'
    UINT32 = 'I'
    UINT64 = 'Q'

    @classmethod
    def format(cls, cpp_types):
        """
        Given a Cpp type list, and return a python struct format string.

        Args:
            cpp_types (list): The cpp type list that should be a member of StructType.

        Returns:
            str, a python struct format string.

        Example:
            >>> cpp_typs = [StructType.UINT16, StructType.UINT32, StructType.UINT32]
            >>> ret = StructType.format(cpp_typs)
            >>> print(ret)
            ... 'HII'
        """
        return ''.join([member.value for member in cpp_types])

    @classmethod
    def sizeof(cls, cpp_type):
        """Given a Cpp type list or a StructType value, and return a python struct format size."""
        size_map = dict(
            CHAR=1,
            UCHAR=1,
            UINT16=2,
            UINT32=4,
            UINT64=8
        )
        if isinstance(cpp_type, StructType):
            return size_map[cpp_type.name]

        size = 0
        for member in cpp_type:
            if isinstance(member, list):
                size += cls.sizeof(member)
            else:
                size += size_map[member.name]
        return size
