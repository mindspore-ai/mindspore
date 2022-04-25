# Copyright 2022 Huawei Technologies Co., Ltd
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

"""ReduceStd op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

reduce_std_op_info = TBERegOp("ReduceStd") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("reduce_std.so") \
    .compute_cost(10) \
    .kernel_name("reduce_std") \
    .partial_flag(True) \
    .attr("axis", "optional", "listInt", "all", "()") \
    .attr("unbiased", "optional", "bool", "all", "true") \
    .attr("keep_dims", "optional", "bool", "all", "false") \
    .input(0, "input_x", False, "required", "all") \
    .output(0, "output_std", False, "required", "all") \
    .output(1, "output_mean", False, "required", "all") \
    .dtype_format(DataType.F16_NCHW, DataType.F16_NCHW, DataType.F16_NCHW) \
    .dtype_format(DataType.F16_NHWC, DataType.F16_NHWC, DataType.F16_NHWC) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(reduce_std_op_info)
def _reduce_std_tbe():
    """ReduceStd TBE register"""
    return
