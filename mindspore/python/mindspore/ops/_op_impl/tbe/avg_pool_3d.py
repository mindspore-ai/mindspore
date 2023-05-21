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

"""AvgPool3D op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

avg_pool_3d_op_info = TBERegOp("AvgPool3D") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("avg_pool3d_d.so") \
    .compute_cost(10) \
    .kernel_name("avg_pool3d_d") \
    .partial_flag(True) \
    .attr("kernel_size", "required", "listInt", "all", "[]") \
    .attr("strides", "required", "listInt", "all", "[]") \
    .attr("pad_list", "required", "listInt", "all", "[]") \
    .attr("ceil_mode", "optional", "bool", "all") \
    .attr("count_include_pad", "optional", "bool", "all") \
    .attr("divisor_override", "optional", "int", "all", '0') \
    .attr("format", "optional", "str", "all", 'NCDHW') \
    .input(0, "x", False, "required", "all") \
    .input(1, "filter", False, "optional", "all") \
    .input(2, "multiplier", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F16_FRACTAL_Z_3D, DataType.F16_NDC1HWC0, DataType.F16_NDC1HWC0) \
    .get_op_info()


@op_info_register(avg_pool_3d_op_info)
def _avg_pool_3d_tbe():
    """AvgPool3D TBE register"""
    return
