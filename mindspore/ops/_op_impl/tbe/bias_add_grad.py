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

"""BiasAddGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bias_add_grad_op_info = TBERegOp("BiasAddGrad") \
    .fusion_type("COMMREDUCE") \
    .async_flag(False) \
    .binfile_name("bias_add_grad.so") \
    .compute_cost(10) \
    .kernel_name("bias_add_grad") \
    .partial_flag(True) \
    .attr("format", "required", "str", "all") \
    .input(0, "output_backprop", False, "required", "all") \
    .output(0, "output", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_FracNZ, DataType.F32_Default) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_FracNZ, DataType.F32_NHWC) \
    .dtype_format(DataType.F16_Default, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_NHWC) \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F16_Default) \
    .dtype_format(DataType.F32_NDC1HWC0, DataType.F32_Default) \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_NDC1HWC0, DataType.F32_NHWC) \
    .dtype_format(DataType.F16_FRACTAL_Z_3D, DataType.F16_Default) \
    .dtype_format(DataType.F32_FRACTAL_Z_3D, DataType.F32_Default) \
    .dtype_format(DataType.F16_FRACTAL_Z_3D, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_FRACTAL_Z_3D, DataType.F32_NHWC) \
    .get_op_info()


@op_info_register(bias_add_grad_op_info)
def _bias_add_grad_tbe():
    """BiasAddGrad TBE register"""
    return
