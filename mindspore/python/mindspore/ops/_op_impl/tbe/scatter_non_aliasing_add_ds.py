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

"""ScatterNonAliasingAdd op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

scatter_non_aliasing_add_ds_op_info = TBERegOp("ScatterNonAliasingAdd") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("scatter_non_aliasing_add.so") \
    .compute_cost(10) \
    .kernel_name("scatter_non_aliasing_add") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .input(0, "input_x", False, "required", "all") \
    .input(1, "indices", False, "required", "all") \
    .input(2, "updates", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(scatter_non_aliasing_add_ds_op_info)
def _scatter_non_aliasing_add_ds_tbe():
    """ScatterNonAliasingAdd TBE register"""
    return
