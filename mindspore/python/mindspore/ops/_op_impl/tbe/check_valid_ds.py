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

"""CheckValid op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

check_valid_op_info = TBERegOp("CheckValid") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("check_valid.so") \
    .compute_cost(10) \
    .kernel_name("check_valid") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .input(0, "bbox_tensor", False, "required", "all") \
    .input(1, "img_tas", False, "required", "all") \
    .output(0, "valid_tensor", False, "required", "all") \
    .op_pattern("broadcast") \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.I8_None) \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.BOOL_None) \
    .get_op_info()


@op_info_register(check_valid_op_info)
def _check_valid_ds_tbe():
    """CheckValid TBE register"""
    return
