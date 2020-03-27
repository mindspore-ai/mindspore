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
import mindspore._c_dataengine as cde

from .validators import check_num_classes, check_de_type
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
