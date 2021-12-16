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

"""ReluV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

relu_v2_op_info = TBERegOp("ReLUV2") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("relu_v2.so") \
    .compute_cost(10) \
    .kernel_name("relu_v2") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "mask", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.U8_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.U8_Default) \
    .dtype_format(DataType.I32_5HD, DataType.I32_5HD, DataType.U8_Default) \
    .dtype_format(DataType.I8_5HD, DataType.I8_5HD, DataType.U8_Default) \
    .dtype_format(DataType.U8_5HD, DataType.U8_5HD, DataType.U8_Default) \
    .get_op_info()


@op_info_register(relu_v2_op_info)
def _relu_v2_ds_tbe():
    """ReluV2 TBE register"""
    return
