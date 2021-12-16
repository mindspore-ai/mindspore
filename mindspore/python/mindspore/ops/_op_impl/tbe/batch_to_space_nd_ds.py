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

"""BatchToSpaceND op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

batch_to_space_nd_ds_op_info = TBERegOp("BatchToSpaceND") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batch_to_space_nd_d.so") \
    .compute_cost(10) \
    .kernel_name("batch_to_space_nd_d") \
    .partial_flag(True) \
    .dynamic_shape(True)\
    .attr("block_shape", "required", "listInt", "all") \
    .attr("crops", "required", "listListInt", "all") \
    .input(0, "x", False, "required", "all", reshape_type="NH") \
    .output(0, "y", False, "required", "all", reshape_type="NH") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(batch_to_space_nd_ds_op_info)
def _batch_to_space_nd_ds_tbe():
    """BatchToSpaceND TBE register"""
    return
