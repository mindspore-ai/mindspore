# Copyright 2023 Huawei Technologies Co., Ltd
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

"""NPUClearFloatStatusV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

npu_clear_float_status_v2_op_info = TBERegOp("NPUClearFloatStatusV2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("n_p_u_clear_float_status_v2.so") \
    .compute_cost(10) \
    .kernel_name("n_p_u_clear_float_status_v2") \
    .partial_flag(True) \
    .input(0, "addr", False, "required", "all") \
    .output(0, "data", False, "required", "all") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(npu_clear_float_status_v2_op_info)
def _npu_clear_float_status_v2_tbe():
    """NPUClearFloatStatusV2 TBE register"""
    return
