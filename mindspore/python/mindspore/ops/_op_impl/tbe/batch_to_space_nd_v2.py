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

"""BatchToSpaceNDV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

batch_to_space_nd_v2_op_info = TBERegOp("BatchToSpaceNDV2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batch_to_space_nd.so") \
    .compute_cost(10) \
    .kernel_name("batch_to_space_nd") \
    .partial_flag(True) \
    .need_check_supported(True) \
    .dynamic_compile_static(True) \
    .dynamic_shape(True) \
    .input(0, "x", False, "required", "all", reshape_type="NHC") \
    .input(1, "block_shape", False, "required", "all", "optional") \
    .input(2, "crops", False, "required", "all", "optional") \
    .output(0, "y", False, "required", "all", reshape_type="NHC") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(batch_to_space_nd_v2_op_info)
def _batch_to_space_nd_v2_tbe():
    """BatchToSpaceND TBE register"""
    return
