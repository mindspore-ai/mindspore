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

"""Lerp op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lerp_op_info = TBERegOp("Lerp") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("lerp.so") \
    .compute_cost(10) \
    .kernel_name("lerp") \
    .partial_flag(True) \
    .input(0, "start", False, "required", "all") \
    .input(1, "end", False, "required", "all") \
    .input(2, "weight", False, "required", "all") \
    .output(0, "output", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(lerp_op_info)
def _lerp_tbe():
    """Lerp TBE register"""
    return
