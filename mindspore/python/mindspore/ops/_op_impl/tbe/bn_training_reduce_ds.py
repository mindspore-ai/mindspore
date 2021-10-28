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

"""BatchNorm op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bn_training_reduce_ds_op_info = TBERegOp("BNTrainingReduce") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("bn_training_reduce.so") \
    .compute_cost(10) \
    .kernel_name("bn_training_reduce") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .input(0, "x", False, "required", "all", reshape_type="NC") \
    .output(0, "sum", False, "required", "all") \
    .output(1, "square_sum", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.F16_None, DataType.F32_None, DataType.F32_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bn_training_reduce_ds_op_info)
def _bn_training_reduce_ds_tbe():
    """BNTrainingReduce TBE register"""
    return
