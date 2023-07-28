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
import struct
from enum import Enum


class StructType(Enum):
    """Define cpp struct type, value is python struct type."""
    CHAR = 'c'
    UINT8 = 'B'
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
        """Given a Cpp type list or dict or a StructType value, and return a python struct format size."""
        size_map = dict(
            CHAR=1,
            UINT8=1,
            UINT16=2,
            UINT32=4,
            UINT64=8
        )
        if isinstance(cpp_type, dict):
            cpp_type = cpp_type.values()
        if isinstance(cpp_type, StructType):
            return size_map.get(cpp_type.name)

        size = 0
        for member in cpp_type:
            if isinstance(member, list):
                size += cls.sizeof(member)
            else:
                size += size_map.get(member.name)
        return size

    @classmethod
    def unpack_binary_data(cls, data_struct, binary_data, special_process_func=None):
        """
        Parse the binary data to get the unpacked data.

        Args:
            data_struct (dict): Key is the data name, value is StructType.
            binary_data (str): This value should be a binary string.
            special_func (Callable): This is a callable function,
                the arguments are item_binary_data, data_name, data_type, unpacked_data.
                This function should return a tuple, first value is unpacked data, second value is success flag,
                If data can not unpack, this function should return None.

        Returns:
            dict, key is data name, value is actual value.

        Example:
            >>> ret = StructType.unpack_binary_data({'op_id': StructType.UINT8}, b'a')
            >>> print(ret)
            {'op_name': 97}
            >>> # special_process_func example
            >>> def handle_tensor_number(binary_data, data_size, cursor, item_name, iten_type)
            ...     if data_name == 'tensorNum':
            ...         tensor_num_struct = data_type[0]
            ...         size = StructType.sizeof(tensor_num_struct)
            ...          unpack_data = struct.unpack(tensor_num_struct.value, binary_data[cursor:cursor + size])[0]
            ...          return unpack_data, True
            ...      return None, False
            ...
            >>> data_struct = {'tensorNum': [StructType.UINT32]}
            >>> ret = StructType.unpack_binary_data(data_struct, b'1101', special_func=handle_tensor_number)
            >>> print(ret)
        """
        unpacked_data = {}
        cursor = 0
        for name, data_type in data_struct.items():
            data_size = StructType.sizeof(data_type)
            if special_process_func:
                unpack_data, success = special_process_func(binary_data[cursor:cursor + data_size], name,
                                                            data_type, unpacked_data)
                if success:
                    cursor += data_size
                    unpacked_data[name] = unpack_data
                    continue

            unpack_data = struct.unpack(data_type.value, binary_data[cursor: cursor + data_size])[0]
            cursor += data_size
            unpacked_data[name] = unpack_data
        return unpacked_data
