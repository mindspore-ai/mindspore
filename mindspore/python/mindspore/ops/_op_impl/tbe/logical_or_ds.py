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

"""LogicalOr op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

logical_or_op_info = TBERegOp("LogicalOr") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("logical_or.so") \
    .compute_cost(10) \
    .kernel_name("logical_or") \
    .dynamic_shape(True) \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .output(0, "y", True, "required", "all") \
    .op_pattern("broadcast") \
    .dtype_format(DataType.BOOL_None, DataType.BOOL_None, DataType.BOOL_None) \
    .get_op_info()


@op_info_register(logical_or_op_info)
def _logical_or_ds_tbe():
    """LogicalOr TBE register"""
    return
