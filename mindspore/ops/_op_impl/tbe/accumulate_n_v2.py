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

"""AccumulateNV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

accumulate_n_v2_op_info = TBERegOp("AccumulateNV2") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("accumulate_n_v2.so") \
    .compute_cost(10) \
    .kernel_name("accumulate_n_v2") \
    .partial_flag(True) \
    .attr("n", "required", "int", "all") \
    .input(0, "x", False, "dynamic", "all") \
    .output(0, "y", False, "required", "all") \
    .op_pattern("broadcast") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default) \
    .get_op_info()


@op_info_register(accumulate_n_v2_op_info)
def _accumulate_n_v2_tbe():
    """AccumulateNV2 TBE register"""
    return
