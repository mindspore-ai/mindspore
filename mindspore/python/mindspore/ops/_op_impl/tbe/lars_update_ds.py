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

"""LarsUpdate op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lars_update_op_info = TBERegOp("LARSUpdate") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("lars_v2_update.so") \
    .compute_cost(10) \
    .kernel_name("lars_v2_update") \
    .dynamic_shape(True) \
    .partial_flag(True) \
    .attr("hyperpara", "optional", "float", "all") \
    .attr("epsilon", "optional", "float", "all") \
    .attr("use_clip", "optional", "bool", "all") \
    .input(0, "w", False, "required", "all") \
    .input(1, "g", False, "required", "all") \
    .input(2, "w_square_sum", False, "required", "all") \
    .input(3, "g_square_sum", False, "required", "all") \
    .input(4, "weight_decay", False, "required", "all") \
    .input(5, "learning_rate", False, "required", "all") \
    .output(0, "g_new", False, "required", "all") \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_FracZ) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_C1HWNCoC0) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(lars_update_op_info)
def _lars_update_ds_tbe():
    """LarsUpdate TBE register"""
    return
