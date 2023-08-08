# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
"""Operator argument data type cast function."""
from enum import Enum


class TypeCastKind(Enum):
    INT_TO_TUPLE = 1
    INT_OR_TUPLE_TO_LIST = 2


def type_it(src_data, cast_type):
    """
    cast operator argument data type.
    """
    if cast_type == TypeCastKind.INT_TO_TUPLE:
        if isinstance(src_data, tuple):
            return src_data

        if isinstance(src_data, int):
            return (src_data,)

        raise TypeError(f'{src_data} is the wrong data type.')

    if cast_type == TypeCastKind.INT_OR_TUPLE_TO_LIST:
        if isinstance(src_data, list):
            return src_data

        if isinstance(src_data, int):
            return [
                src_data,
            ]

        if isinstance(src_data, tuple):
            dst_list = [item for item in src_data]
            return dst_list

        raise TypeError(f'{src_data} is the wrong data type.')

    raise TypeError("Unsupported type cast")
