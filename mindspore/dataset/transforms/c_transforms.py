# Copyright 2019 Huawei Technologies Co., Ltd
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
"""
This module c_transforms provides common operations, including OneHotOp and TypeCast.
"""
from enum import IntEnum

import mindspore.common.dtype as mstype
import mindspore._c_dataengine as cde

import numpy as np

from .validators import check_num_classes, check_de_type, check_fill_value, check_slice_op, check_mask_op
from ..core.datatypes import mstype_to_detype


class OneHot(cde.OneHotOp):
    """
    Tensor operation to apply one hot encoding.

    Args:
        num_classes (int): Number of classes of the label.
    """

    @check_num_classes
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes)


class Fill(cde.FillOp):
    """
    Tensor operation to create a tensor filled with passed scalar value.
    The output tensor will have the same shape and type as the input tensor.

    Args:
        fill_value (python types (str, int, float, or bool)) : scalar value
            to fill created tensor with.
    """

    @check_fill_value
    def __init__(self, fill_value):
        super().__init__(cde.Tensor(np.array(fill_value)))


class TypeCast(cde.TypeCastOp):
    """
    Tensor operation to cast to a given MindSpore data type.

    Args:
        data_type (mindspore.dtype): mindspore.dtype to be casted to.
    """

    @check_de_type
    def __init__(self, data_type):
        data_type = mstype_to_detype(data_type)
        self.data_type = str(data_type)
        super().__init__(data_type)


class Slice(cde.SliceOp):
    """
    Slice operation to extract a tensor out using the given n slices.

    The functionality of Slice is similar to NumPy indexing feature.

    (Currently only rank 1 Tensors are supported)

    Args:
     *slices: Maximum n number of objects to slice a tensor of rank n.
         One object in slices can be one of:
             1.  int: slice this index only. Negative index is supported.
             2.  slice object: slice the generated indices from the slice object. Similar to `start:stop:step`.
             3.  None: slice the whole dimension. Similar to `:` in python indexing.
             4.  Ellipses ...: slice all dimensions between the two slices.
    Examples:
     >>> # Data before
     >>> # |   col   |
     >>> # +---------+
     >>> # | [1,2,3] |
     >>> # +---------|
     >>> data = data.map(operations=Slice(slice(1,3))) # slice indices 1 and 2 only
     >>> # Data after
     >>> # |    col     |
     >>> # +------------+
     >>> # |    [1,2]   |
     >>> # +------------|
    """

    @check_slice_op
    def __init__(self, *slices):
        dim0 = slices[0]
        if isinstance(dim0, int):
            dim0 = [dim0]
        elif dim0 is None:
            dim0 = True
        elif isinstance(dim0, slice):
            dim0 = (dim0.start, dim0.stop, dim0.step)
        elif dim0 is Ellipsis:
            dim0 = True
        super().__init__(dim0)


class Relational(IntEnum):
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5


DE_C_RELATIONAL = {Relational.EQ: cde.RelationalOp.EQ,
                   Relational.NE: cde.RelationalOp.NE,
                   Relational.GT: cde.RelationalOp.GT,
                   Relational.GE: cde.RelationalOp.GE,
                   Relational.LT: cde.RelationalOp.LT,
                   Relational.LE: cde.RelationalOp.LE}


class Mask(cde.MaskOp):
    """
    Mask content of the input tensor with the given predicate.
    Any element of the tensor that matches the predicate will be evaluated to True, otherwise False.
    Args:
        operator (Relational): One of the relational operator EQ, NE LT, GT, LE or GE
        constant (python types (str, int, float, or bool): constant to be compared to.
            Constant will be casted to the type of the input tensor
        dtype (optional, mindspore.dtype): type of the generated mask. Default to bool
    Examples:
        >>> # Data before
        >>> # |  col1   |
        >>> # +---------+
        >>> # | [1,2,3] |
        >>> # +---------+
        >>> data = data.map(operations=Mask(Relational.EQ, 2))
        >>> # Data after
        >>> # |       col1         |
        >>> # +--------------------+
        >>> # | [False,True,False] |
        >>> # +--------------------+
    """

    @check_mask_op
    def __init__(self, operator, constant, dtype=mstype.bool_):
        dtype = mstype_to_detype(dtype)
        constant = cde.Tensor(np.array(constant))
        super().__init__(DE_C_RELATIONAL[operator], constant, dtype)
