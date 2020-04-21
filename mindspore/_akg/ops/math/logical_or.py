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

"""operator dsl function: logical_or"""
import _akg.tvm
import _akg.topi
from _akg.utils import validation_check as vc_util

@vc_util.check_input_type(_akg.tvm.tensor.Tensor, _akg.tvm.tensor.Tensor)
def logical_or(input1, input2):
    """
    Compute logical_or of input1 and input2.

    Args:
        input1 (tvm.tensor.Tensor): Tensor.
        input2 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor. LogicalOr of input1 and input2.
    """

    vc_util.elemwise_dtype_check(input1.dtype, input2.dtype)

    shape1 = [x.value for x in input1.shape]
    shape2 = [x.value for x in input2.shape]
    vc_util.check_shape(shape1)
    vc_util.check_shape(shape2)

    res = _akg.topi.logical_or(input1, input2)
    return res
