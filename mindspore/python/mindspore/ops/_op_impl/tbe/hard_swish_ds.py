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

"""Hard Swish op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

hard_swish_op_info = TBERegOp("HSwish") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("hard_swish.so") \
    .compute_cost(10) \
    .kernel_name("hard_swish") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .op_pattern("formatAgnostic") \
    .dynamic_shape(True) \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(hard_swish_op_info)
def _hard_swish_ds_tbe():
    """Hard Swish TBE register"""
    return
