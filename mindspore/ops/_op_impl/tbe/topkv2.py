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

"""TopKV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

top_k_v2_op_info = TBERegOp("TopKV2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("top_k_v2.so") \
    .compute_cost(10) \
    .kernel_name("top_k_v2") \
    .partial_flag(True) \
    .attr("k", "required", "int", "all")\
    .attr("sorted", "required", "bool", "all")\
    .input(0, "x", False, "required", "all") \
    .input(1, "input_indices", False, "optional", "all") \
    .output(0, "values", False, "required", "all") \
    .output(1, "indices", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(top_k_v2_op_info)
def _topk_v2_tbe():
    """TopKV2 TBE register"""
    return
