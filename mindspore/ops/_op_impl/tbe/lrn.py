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

"""LRN op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lrn_op_info = TBERegOp("LRN") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("lrn.so") \
    .compute_cost(10) \
    .kernel_name("lrn") \
    .partial_flag(True) \
    .attr("depth_radius", "optional", "int", "all", "5") \
    .attr("bias", "optional", "float", "all", "1.0") \
    .attr("alpha", "optional", "float", "all", "1.0") \
    .attr("beta", "optional", "float", "all", "0.5") \
    .attr("norm_region", "optional", "str", "all", "ACROSS_CHANNELS") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(lrn_op_info)
def _lrn_tbe():
    """LRN TBE register"""
    return
