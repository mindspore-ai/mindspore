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

"""operator dsl function: cast"""
import _akg.tvm
import _akg.topi
from _akg.utils import validation_check as vc_util


@vc_util.check_input_type(_akg.tvm.tensor.Tensor, str)
def cast(data, dst_type):
    """
    cast data to target type.

    Args:
        data (tvm.tensor.Tensor): Tensor to be casted.
        dst_type (str): target cast type.

    Returns:
        tvm.tensor.Tensor, type is dst_type.
    """
    vc_util.check_shape(data.shape)
    out = _akg.topi.cast(data, dst_type)

    return out
