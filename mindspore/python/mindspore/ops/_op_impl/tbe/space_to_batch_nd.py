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

"""SpaceToBatchND op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

space_to_batch_nd_op_info = TBERegOp("SpaceToBatchNDD") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("space_to_batch_nd_d.so") \
    .compute_cost(10) \
    .kernel_name("space_to_batch_nd_d") \
    .partial_flag(True) \
    .attr("block_shape", "required", "listInt", "all", "[]") \
    .attr("paddings", "required", "listListInt", "all", "[[]]") \
    .input(0, "x", False, "required", "all", reshape_type="NH") \
    .output(0, "y", False, "required", "all", reshape_type="NH") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(space_to_batch_nd_op_info)
def _space_to_batch_nd_tbe():
    """SpaceToBatchND TBE register"""
    return
