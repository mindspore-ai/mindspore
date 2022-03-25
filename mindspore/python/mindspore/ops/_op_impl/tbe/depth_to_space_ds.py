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

"""DepthToSpace op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

depth_to_space_ds_op_info = TBERegOp("DepthToSpace") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("depth_to_space.so") \
    .compute_cost(10) \
    .kernel_name("depth_to_space") \
    .partial_flag(True) \
    .dynamic_compile_static(True) \
    .dynamic_shape(True) \
    .attr("block_size", "required", "int", "all") \
    .attr("mode", "optional", "str", "all", "DCR") \
    .attr("data_format", "optional", "str", "all", "NCHW") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_NCHW, DataType.F16_NCHW) \
    .dtype_format(DataType.F32_NCHW, DataType.F32_NCHW) \
    .dtype_format(DataType.I8_NCHW, DataType.I8_NCHW) \
    .dtype_format(DataType.I16_NCHW, DataType.I16_NCHW) \
    .dtype_format(DataType.I32_NCHW, DataType.I32_NCHW) \
    .dtype_format(DataType.I64_NCHW, DataType.I64_NCHW) \
    .dtype_format(DataType.U8_NCHW, DataType.U8_NCHW) \
    .dtype_format(DataType.U16_NCHW, DataType.U16_NCHW) \
    .dtype_format(DataType.U32_NCHW, DataType.U32_NCHW) \
    .dtype_format(DataType.U64_NCHW, DataType.U64_NCHW) \
    .get_op_info()


@op_info_register(depth_to_space_ds_op_info)
def _depth_to_space_ds_tbe():
    """DepthToSpace TBE register"""
    return
