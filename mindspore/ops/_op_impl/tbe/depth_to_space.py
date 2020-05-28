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

"""DepthToSpace op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

depth_to_space_op_info = TBERegOp("DepthToSpace") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("depth_to_space.so") \
    .compute_cost(10) \
    .kernel_name("depth_to_space") \
    .partial_flag(True) \
    .attr("block_size", "required", "int", "all") \
    .attr("data_format", "optional", "str", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_NHWC, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC) \
    .dtype_format(DataType.I8_NHWC, DataType.I8_NHWC) \
    .dtype_format(DataType.I16_NHWC, DataType.I16_NHWC) \
    .dtype_format(DataType.I32_NHWC, DataType.I32_NHWC) \
    .dtype_format(DataType.I64_NHWC, DataType.I64_NHWC) \
    .dtype_format(DataType.U8_NHWC, DataType.U8_NHWC) \
    .dtype_format(DataType.U16_NHWC, DataType.U16_NHWC) \
    .dtype_format(DataType.U32_NHWC, DataType.U32_NHWC) \
    .dtype_format(DataType.U64_NHWC, DataType.U64_NHWC) \
    .get_op_info()


@op_info_register(depth_to_space_op_info)
def _depth_to_space_tbe():
    """DepthToSpace TBE register"""
    return
