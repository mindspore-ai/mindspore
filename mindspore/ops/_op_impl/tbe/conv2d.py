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

"""Conv2D op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

conv2d_op_info = TBERegOp("Conv2D") \
    .fusion_type("CONVLUTION") \
    .async_flag(False) \
    .binfile_name("conv2d.so") \
    .compute_cost(10) \
    .kernel_name("conv2d") \
    .partial_flag(True) \
    .is_dynamic_format(True) \
    .attr("stride", "required", "listInt", "all") \
    .attr("pad_list", "required", "listInt", "all") \
    .attr("dilation", "required", "listInt", "all") \
    .attr("groups", "optional", "int", "all") \
    .attr("format", "optional", "str", "all") \
    .attr("offset_x", "optional", "int", "all", "0") \
    .input(0, "x", False, "required", "all") \
    .input(1, "filter", False, "required", "all") \
    .input(2, "bias", False, "optional", "all") \
    .input(3, "offset_w", False, "optional", "all") \
    .output(0, "y", True, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.F16_None, DataType.I8_None, DataType.F16_None) \
    .dtype_format(DataType.I8_None, DataType.I8_None, DataType.I32_None, DataType.I8_None, DataType.I32_None) \
    .get_op_info()


@op_info_register(conv2d_op_info)
def _conv2d_tbe():
    """Conv2D TBE register"""
    return
