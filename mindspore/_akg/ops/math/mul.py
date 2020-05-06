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

"""operator dsl function: mul"""
import _akg.topi
from _akg.utils import validation_check as vc_util


@vc_util.check_input_type(_akg.tvm.tensor.Tensor, _akg.tvm.tensor.Tensor)
def mul(l_input, r_input):
    """
    Calculate x * y element-wise.

    Note:
        mul supports broadcasting.

    Args:
        l_input (tvm.tensor.Tensor): Tensor.
        r_input (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor, has the same type as l_input and r_input.
    """
    shape1 = [x.value for x in l_input.shape]
    shape2 = [x.value for x in r_input.shape]
    vc_util.check_shape(shape1)
    vc_util.check_shape(shape2)
    vc_util.auto_broadcast_check(shape1, shape2)
    vc_util.elemwise_dtype_check(l_input.dtype, r_input.dtype)
    output = _akg.topi.multiply(l_input, r_input)

    return output
