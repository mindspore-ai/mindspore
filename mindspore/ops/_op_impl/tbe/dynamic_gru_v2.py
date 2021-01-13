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

"""DynamicGRUV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

dynamic_gru_v2_op_info = TBERegOp("DynamicGRUV2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("dynamic_gru_v2.so") \
    .compute_cost(10) \
    .kernel_name("dynamic_gru_v2") \
    .attr("direction", "optional", "str", "all", "UNIDIRECTIONAL") \
    .attr("cell_depth", "optional", "int", "all", "1") \
    .attr("keep_prob", "optional", "float", "all", "1") \
    .attr("cell_clip", "optional", "float", "all", "-1") \
    .attr("num_proj", "optional", "int", "all", "0") \
    .attr("time_major", "optional", "bool", "all", "true") \
    .attr("activation", "optional", "str", "all", "tanh") \
    .attr("gate_order", "optional", "str", "all", "rzh") \
    .attr("reset_after", "optional", "bool", "all", "true") \
    .attr("is_training", "optional", "bool", "all", "true") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .input(1, "weight_input", False, "required", "all", reshape_type="CN") \
    .input(2, "weight_hidden", False, "required", "all", reshape_type="CN") \
    .input(3, "bias_input", False, "optional", "all") \
    .input(4, "bias_hidden", False, "optional", "all") \
    .input(5, "seq_length", False, "optional", "all") \
    .input(6, "init_h", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "output_h", False, "required", "all") \
    .output(2, "update", False, "optional", "all") \
    .output(3, "reset", False, "optional", "all") \
    .output(4, "new", False, "optional", "all") \
    .output(5, "hidden_new", False, "optional", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F32_Default,
                  DataType.F32_Default, DataType.None_Default, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F32_Default,
                  DataType.None_Default, DataType.None_Default, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.None_Default,
                  DataType.F32_Default, DataType.None_Default, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.None_Default,
                  DataType.None_Default, DataType.None_Default, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_Default,
                  DataType.F16_Default, DataType.None_Default, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_Default,
                  DataType.None_Default, DataType.None_Default, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.None_Default,
                  DataType.F16_Default, DataType.None_Default, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.None_Default,
                  DataType.None_Default, DataType.None_Default, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F32_Default,
                  DataType.F32_Default, DataType.I32_Default, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_Default,
                  DataType.F16_Default, DataType.I32_Default, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(dynamic_gru_v2_op_info)
def _dynamic_gru_v2_tbe():
    """DynamicGRUV2 TBE register"""
    return
