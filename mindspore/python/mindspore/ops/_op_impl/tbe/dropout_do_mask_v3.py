# Copyright 2021 Huawei Technologies Co., Ltd
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

"""DropoutDoMaskV3 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

drop_out_do_mask_v3_op_info = TBERegOp("DropoutDoMaskV3") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("drop_out_do_mask_v3_d.so") \
    .compute_cost(10) \
    .kernel_name("drop_out_do_mask_v3_d") \
    .partial_flag(True) \
    .attr("keep_prob", "required", "float", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "mask", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.U8_Default, DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_Default, DataType.U8_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.U8_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(drop_out_do_mask_v3_op_info)
def _dropout_do_mask_v3_tbe():
    """DropoutDoMaskV3 TBE register"""
    return
