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

"""GRUV2HiddenGradCell op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

gru_v2_hidden_grad_cell_op_info = TBERegOp("GRUV2HiddenGradCell") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("gru_v2_hidden_grad_cell.so") \
    .compute_cost(10) \
    .kernel_name("gru_v2_hidden_grad_cell") \
    .attr("t_state", "optional", "int", "all", "0") \
    .attr("gate_order", "optional", "str", "all", "rzh") \
    .partial_flag(True) \
    .input(0, "dh_pre_t", False, "required", "all") \
    .input(1, "h", False, "required", "all") \
    .input(2, "dy", False, "required", "all") \
    .input(3, "dh", False, "required", "all") \
    .input(4, "update", False, "required", "all") \
    .input(5, "reset", False, "required", "all") \
    .input(6, "new", False, "required", "all") \
    .input(7, "hidden_new", False, "required", "all") \
    .input(8, "seq_mask", False, "optional", "all") \
    .output(0, "dh_preh", False, "required", "all") \
    .output(1, "dgate_h", False, "required", "all") \
    .output(2, "dnt_x", False, "required", "all") \
    .dtype_format(DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(gru_v2_hidden_grad_cell_op_info)
def _gru_v2_hidden_grad_cell_tbe():
    """DynamicGRUV2GradCell TBE register"""
    return
