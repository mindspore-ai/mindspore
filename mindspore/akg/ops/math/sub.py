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

"""operator dsl function: sub"""
import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def sub(data1, data2):
    """
    Computes data1 - data2 elementwise, broadcast is supported.

    Args:
        data1 (tvm.tensor.Tensor): Tensor.
        data2 (tvm.tensor.Tensor): Tensor of same type as data1, if shape(data2) != shape(data1), broadcast will happen.

    Returns:
        tvm.tensor.Tensor, subtracted result, with same type as input tensors and broadcasted shape of data1 and data2.
    """
    vc_util.elemwise_dtype_check(data1.dtype, data2.dtype)
    vc_util.check_shape(data1.shape)
    vc_util.check_shape(data2.shape)
    vc_util.auto_broadcast_check(data1.shape, data2.shape)

    res = akg.topi.subtract(data1, data2)

    return res
