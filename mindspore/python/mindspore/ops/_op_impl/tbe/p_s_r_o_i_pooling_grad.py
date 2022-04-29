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

"""PSROIPoolingGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

p_s_r_o_i_pooling_grad_op_info = TBERegOp("PSROIPoolingGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("p_s_r_o_i_pooling_grad_v2_d.so") \
    .compute_cost(10) \
    .kernel_name("p_s_r_o_i_pooling_grad_v2_d") \
    .partial_flag(True) \
    .attr("output_dim", "required", "int", "all") \
    .attr("group_size", "required", "int", "all") \
    .attr("spatial_scale", "required", "float", "all") \
    .attr("input_size", "required", "listInt", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "rois", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.F32_Default, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(p_s_r_o_i_pooling_grad_op_info)
def _p_s_r_o_i_pooling_grad_tbe():
    """PSROIPoolingGrad TBE register"""
    return
