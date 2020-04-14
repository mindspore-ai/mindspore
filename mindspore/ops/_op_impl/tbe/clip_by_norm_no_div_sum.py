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

"""ClipByNormNoDivSum op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

clip_by_norm_no_div_sum_op_info = TBERegOp("ClipByNormNoDivSum") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("clip_by_norm_no_div_sum.so") \
    .compute_cost(10) \
    .kernel_name("clip_by_norm_no_div_sum") \
    .partial_flag(True) \
    .input(0, "input_x", False, "required", "all") \
    .input(1, "input1", False, "required", "all") \
    .input(2, "input2", False, "required", "all") \
    .input(3, "input3", False, "required", "all") \
    .output(0, "output_y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .get_op_info()


@op_info_register(clip_by_norm_no_div_sum_op_info)
def _clip_by_norm_no_div_sum_tbe():
    """ClipByNormNoDivSum TBE register"""
    return
