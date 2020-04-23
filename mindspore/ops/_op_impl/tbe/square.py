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
# ============================================================================

"""Square op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

square_op_info = TBERegOp("Square") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("square.so") \
    .compute_cost(10) \
    .kernel_name("square") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_5HD, DataType.I32_5HD) \
    .dtype_format(DataType.I32_NHWC, DataType.I32_NHWC) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F16_NHWC, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC) \
    .get_op_info()


@op_info_register(square_op_info)
def _square_tbe():
    """Square TBE register"""
    return
