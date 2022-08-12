# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Define the data types."""
from __future__ import absolute_import

import numpy as np

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing
import mindspore.common.dtype as mstype


def nptype_to_detype(type_):
    """
    Get de data type corresponding to numpy dtype.

    Args:
        type_ (numpy.dtype): Numpy's dtype.

    Returns:
        The data type of de.
    """
    if not isinstance(type_, np.dtype):
        type_ = np.dtype(type_)

    return {
        np.dtype("bool"): cde.DataType("bool"),
        np.dtype("int8"): cde.DataType("int8"),
        np.dtype("int16"): cde.DataType("int16"),
        np.dtype("int32"): cde.DataType("int32"),
        np.dtype("int64"): cde.DataType("int64"),
        np.dtype("uint8"): cde.DataType("uint8"),
        np.dtype("uint16"): cde.DataType("uint16"),
        np.dtype("uint32"): cde.DataType("uint32"),
        np.dtype("uint64"): cde.DataType("uint64"),
        np.dtype("float16"): cde.DataType("float16"),
        np.dtype("float32"): cde.DataType("float32"),
        np.dtype("float64"): cde.DataType("float64"),
        np.dtype("str"): cde.DataType("string"),
    }.get(type_)


def mstype_to_detype(type_):
    """
    Get de data type corresponding to mindspore dtype.

    Args:
        type_ (mindspore.dtype): MindSpore's dtype.

    Returns:
        The data type of de.
    """
    if not isinstance(type_, typing.Type):
        raise NotImplementedError()

    return {
        mstype.bool_: cde.DataType("bool"),
        mstype.int8: cde.DataType("int8"),
        mstype.int16: cde.DataType("int16"),
        mstype.int32: cde.DataType("int32"),
        mstype.int64: cde.DataType("int64"),
        mstype.uint8: cde.DataType("uint8"),
        mstype.uint16: cde.DataType("uint16"),
        mstype.uint32: cde.DataType("uint32"),
        mstype.uint64: cde.DataType("uint64"),
        mstype.float16: cde.DataType("float16"),
        mstype.float32: cde.DataType("float32"),
        mstype.float64: cde.DataType("float64"),
        mstype.string: cde.DataType("string"),
    }[type_]


def mstypelist_to_detypelist(type_list):
    """
    Get list[de type] corresponding to list[mindspore.dtype].

    Args:
        type_list (list[mindspore.dtype]): a list of MindSpore's dtype.

    Returns:
        The list of de data type.
    """
    for index, _ in enumerate(type_list):
        if type_list[index] is not None:
            type_list[index] = mstype_to_detype(type_list[index])
        else:
            type_list[index] = cde.DataType("")

    return type_list
