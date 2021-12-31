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

"""Conv2DTranspose op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

conv2d_transpose_op_info = TBERegOp("Conv2DTranspose") \
    .fusion_type("CONVOLUTION") \
    .async_flag(False) \
    .binfile_name("conv2d_transpose_d.so") \
    .compute_cost(10) \
    .kernel_name("conv2d_transpose_d") \
    .partial_flag(True) \
    .attr("input_sizes", "required", "listInt", "all") \
    .attr("stride", "required", "listInt", "all") \
    .attr("pad_list", "required", "listInt", "all") \
    .attr("dilation", "optional", "listInt", "all", "1,1,1,1") \
    .attr("groups", "optional", "int", "all", "1") \
    .attr("format", "optional", "str", "all", "NHWC") \
    .attr("output_padding", "optional", "listInt", "all", "0,0,0,0") \
    .attr("offset_x", "optional", "int", "all", "0") \
    .input(0, "x", False, "required", "all") \
    .input(1, "filter", False, "required", "all") \
    .input(2, "bias", False, "optional", "all") \
    .input(3, "offset_w", False, "optional", "all") \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_FracZ, DataType.F16_Default, DataType.I8_Default, DataType.F16_5HD) \
    .dtype_format(DataType.F16_5HD, DataType.F16_FracZ, DataType.F32_Default, DataType.I8_Default, DataType.F32_5HD) \
    .dtype_format(DataType.I8_5HD, DataType.I8_FracZ, DataType.I32_Default, DataType.I8_Default, DataType.I32_5HD) \
    .get_op_info()


@op_info_register(conv2d_transpose_op_info)
def _conv2d_transpose_tbe():
    """Conv2DTranspose TBE register"""
    return
