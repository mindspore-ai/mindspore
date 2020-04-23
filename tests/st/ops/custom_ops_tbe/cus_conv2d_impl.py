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
from tests.st.ops.custom_ops_tbe.conv2d import conv2d
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

cus_conv2D_op_info = TBERegOp("Cus_Conv2D") \
    .fusion_type("CONVLUTION") \
    .async_flag(False) \
    .binfile_name("conv2d.so") \
    .compute_cost(10) \
    .kernel_name("Cus_Conv2D") \
    .partial_flag(True) \
    .attr("stride", "required", "listInt", "all") \
    .attr("pad_list", "required", "listInt", "all") \
    .attr("dilation", "required", "listInt", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "filter", False, "required", "all") \
    .input(2, "bias", False, "optional", "all") \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_FracZ, DataType.F32_Default, DataType.F16_5HD) \
    .get_op_info()


@op_info_register(cus_conv2D_op_info)
def Cus_Conv2D(inputs, weights, bias, outputs, strides, pads, dilations,
               kernel_name="conv2d"):
    conv2d(inputs, weights, bias, outputs, strides, pads, dilations,
           kernel_name)
