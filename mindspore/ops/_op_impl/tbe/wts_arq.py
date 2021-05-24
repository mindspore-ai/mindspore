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

"""WtsARQ op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

wts_arq_op_info = TBERegOp("WtsARQ") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("wts_arq.so") \
    .compute_cost(10) \
    .kernel_name("wts_arq") \
    .partial_flag(True) \
    .attr("num_bits", "optional", "int", "all", "8") \
    .attr("offset_flag", "optional", "bool", "all", "false") \
    .input(0, "w", False, "required", "all") \
    .input(1, "w_min", False, "required", "all") \
    .input(2, "w_max", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(wts_arq_op_info)
def _wts_arq_tbe():
    """WtsARQ TBE register"""
    return
