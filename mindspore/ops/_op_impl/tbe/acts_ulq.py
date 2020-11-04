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

"""ActsULQ op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

acts_ulq_op_info = TBERegOp("ActsULQ") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("acts_ulq.so") \
    .compute_cost(10) \
    .kernel_name("acts_ulq") \
    .partial_flag(True) \
    .attr("fixed_min", "optional", "bool", "all") \
    .attr("num_bits", "optional", "int", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "clamp_min", False, "required", "all") \
    .input(2, "clamp_max", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "clamp_min_mask", False, "required", "all") \
    .output(2, "clamp_max_mask", False, "required", "all") \
    .output(3, "x_clamped_loss", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.BOOL_Default, DataType.BOOL_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.BOOL_Default, DataType.BOOL_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(acts_ulq_op_info)
def _acts_ulq_tbe():
    """ActsULQ TBE register"""
    return
