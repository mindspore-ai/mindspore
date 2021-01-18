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

"""BatchNormGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

batch_norm_grad_op_info = TBERegOp("BatchNormGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batch_norm_grad.so") \
    .compute_cost(10) \
    .kernel_name("batch_norm_grad") \
    .partial_flag(True) \
    .attr("epsilon", "optional", "float", "all") \
    .attr("format", "optional", "str", "all") \
    .attr("is_training", "optional", "bool", "all") \
    .input(0, "y_backprop", False, "required", "all") \
    .input(1, "x", False, "required", "all") \
    .input(2, "scale", False, "required", "all") \
    .input(3, "reserve_space_1", False, "required", "all") \
    .input(4, "reserve_space_2", False, "required", "all") \
    .output(0, "x_backprop", False, "required", "all") \
    .output(1, "scale_backprop", False, "required", "all") \
    .output(2, "offset_backprop", False, "required", "all") \
    .output(3, "reserve_space_4", False, "optional", "all") \
    .output(4, "reserve_space_5", False, "optional", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F16_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F16_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(batch_norm_grad_op_info)
def _batch_norm_grad_tbe():
    """BatchNormGrad TBE register"""
    return
