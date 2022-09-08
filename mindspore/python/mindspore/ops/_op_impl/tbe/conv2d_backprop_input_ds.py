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

"""Conv2DBackpropInput op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

conv2d_backprop_input_op_info = TBERegOp("Conv2DBackpropInput") \
    .fusion_type("CONVOLUTION") \
    .async_flag(False) \
    .binfile_name("conv2d_backprop_input.so") \
    .compute_cost(10) \
    .kernel_name("conv2d_backprop_input") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("stride", "required", "listInt", "all") \
    .attr("pad_list", "required", "listInt", "all") \
    .attr("dilation", "required", "listInt", "all") \
    .attr("groups", "optional", "int", "all", "1") \
    .attr("format", "optional", "str", "all", "NHWC") \
    .input(0, "out_backprop", False, "required", "all") \
    .input(1, "filter", False, "required", "all") \
    .input(2, "input_size", False, "required", "all") \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_FracZ, DataType.I32_Default, DataType.F16_5HD) \
    .dtype_format(DataType.F16_5HD, DataType.F16_FracZ, DataType.I32_Default, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(conv2d_backprop_input_op_info)
def _conv2d_backprop_input_ds_tbe():
    """Conv2DBackpropInput TBE register"""
    return
