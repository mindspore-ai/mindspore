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

"""FusedDbnDw op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

fused_dbn_dw_op_info = TBERegOp("FusedDbnDw") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fused_dbn_dw.so") \
    .compute_cost(10) \
    .kernel_name("fused_dbn_dw") \
    .partial_flag(True) \
    .attr("filter_sizes", "required", "listInt", "all") \
    .attr("stride", "required", "listInt", "all") \
    .attr("pad_list", "required", "listInt", "all") \
    .attr("dilation", "required", "listInt", "all") \
    .attr("groups", "optional", "int", "1") \
    .attr("format", "optional", "str", "NHWC") \
    .attr("epsilon", "optional", "float", "0.0001") \
    .input(0, "x", False, "required", "all") \
    .input(1, "grads", False, "required", "all") \
    .input(2, "x_norm", False, "required", "all") \
    .input(3, "diff_scale", False, "required", "all") \
    .input(4, "diff_offset", False, "required", "all") \
    .input(5, "scale", False, "required", "all") \
    .input(6, "batch_mean", False, "required", "all") \
    .input(7, "batch_variance", False, "required", "all") \
    .output(0, "dbn_y", False, "required", "all") \
    .output(1, "dw_y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F16_5HD, DataType.F32_FracZ) \
    .get_op_info()


@op_info_register(fused_dbn_dw_op_info)
def _fused_dbn_dw_tbe():
    """FusedDbnDw TBE register"""
    return
