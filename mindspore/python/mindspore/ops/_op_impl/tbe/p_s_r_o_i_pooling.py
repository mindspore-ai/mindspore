# Copyright 2022 Huawei Technologies Co., Ltd
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

"""PSROIPooling op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

p_s_r_o_i_pooling_op_info = TBERegOp("PSROIPooling") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("p_s_r_o_i_pooling_v2.so") \
    .compute_cost(10) \
    .kernel_name("p_s_r_o_i_pooling_v2") \
    .partial_flag(True) \
    .attr("output_dim", "required", "int", "all") \
    .attr("group_size", "required", "int", "all") \
    .attr("spatial_scale", "required", "float", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "rois", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_Default, DataType.F16_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_Default, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(p_s_r_o_i_pooling_op_info)
def _p_s_r_o_i_pooling_tbe():
    """PSROIPooling TBE register"""
    return
