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

"""DepthwiseConv2D op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

depthwise_conv2d_op_info = TBERegOp("DepthwiseConv2D") \
    .fusion_type("CONVOLUTION") \
    .async_flag(False) \
    .binfile_name("depthwise_conv2d.so") \
    .compute_cost(10) \
    .kernel_name("depthwise_conv2d") \
    .partial_flag(True) \
    .attr("stride", "required", "listInt", "all", "[]") \
    .attr("dilation", "required", "listInt", "all", "[]") \
    .attr("pad_list", "required", "listInt", "all", "[]") \
    .attr("format", "required", "str", "all") \
    .attr("offset_a", "optional", "int", "all", "0") \
    .input(0, "x", False, "required", "all") \
    .input(1, "filter", False, "required", "all") \
    .input(2, "bias", False, "optional", "all") \
    .input(3, "offset_w", False, "optional", "all") \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_C1HWNCoC0, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_5HD) \
    .get_op_info()


@op_info_register(depthwise_conv2d_op_info)
def _depthwise_conv2d_tbe():
    """DepthwiseConv2D TBE register"""
    return
