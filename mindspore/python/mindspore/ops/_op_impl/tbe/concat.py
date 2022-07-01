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

"""Concat op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

concat_op_info = TBERegOp("Concat") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("concat_d.so") \
    .compute_cost(10) \
    .kernel_name("concat_d") \
    .partial_flag(True) \
    .dynamic_compile_static(True) \
    .dynamic_shape(True) \
    .need_check_supported(True) \
    .attr("axis", "required", "int", "all") \
    .input(0, "input_values", False, "dynamic", "all") \
    .output(0, "output_data", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(concat_op_info)
def _concat_tbe():
    """Concat TBE register"""
    return
