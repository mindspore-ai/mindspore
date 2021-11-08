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

"""LpNorm op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lp_norm_op_info = TBERegOp("LpNorm") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("lp_norm.so") \
    .compute_cost(10) \
    .kernel_name("lp_norm") \
    .partial_flag(True) \
    .attr("p", "optional", "int", "all", "2") \
    .attr("axis", "optional", "listInt", "all") \
    .attr("keep_dims", "optional", "bool", "all", "false") \
    .attr("epsilon", "optional", "float", "all", "1e-12") \
    .input(0, "input", False, "required", "all") \
    .output(0, "output", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(lp_norm_op_info)
def _lp_norm_tbe():
    """LpNorm TBE register"""
    return
