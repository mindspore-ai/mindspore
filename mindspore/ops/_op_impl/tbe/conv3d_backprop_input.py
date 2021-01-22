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

"""Conv3DBackpropInput op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

conv3d_backprop_input_op_info = TBERegOp("Conv3DBackpropInput") \
    .fusion_type("CONVLUTION") \
    .async_flag(False) \
    .binfile_name("conv3d_backprop_input_d.so") \
    .compute_cost(10) \
    .kernel_name("conv3d_backprop_input_d") \
    .partial_flag(True) \
    .attr("input_size", "required", "listInt", "all") \
    .attr("strides", "required", "listInt", "all") \
    .attr("pad_list", "required", "listInt", "all") \
    .attr("dilations", "required", "listInt", "all") \
    .attr("groups", "optional", "int", "all") \
    .attr("format", "optional", "str", "all") \
    .input(0, "filter", False, "required", "all") \
    .input(1, "out_backprop", False, "required", "all") \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_FRACTAL_Z_3D, DataType.F16_NDC1HWC0, DataType.F16_NDC1HWC0) \
    .get_op_info()


@op_info_register(conv3d_backprop_input_op_info)
def _conv3d_backprop_input_tbe():
    """Conv3DBackpropInput TBE register"""
    return
