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

"""BatchNormGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bn_training_update_grad_op_info = TBERegOp("BNTrainingUpdateGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("bn_training_update_grad.so") \
    .compute_cost(10) \
    .kernel_name("bn_training_update_grad") \
    .partial_flag(True) \
    .attr("epsilon", "optional", "float", "all", "0.0001") \
    .input(0, "grads", False, "required", "all", reshape_type="NC") \
    .input(1, "x", False, "required", "all", reshape_type="NC") \
    .input(2, "batch_mean", False, "required", "all") \
    .input(3, "batch_variance", False, "required", "all") \
    .output(0, "diff_scale", False, "required", "all") \
    .output(1, "diff_offset", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.F32_None, DataType.F32_None,
                  DataType.F32_None, DataType.F32_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None, DataType.F32_None,
                  DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bn_training_update_grad_op_info)
def _bn_training_update_grad_tbe():
    """BNTrainingUpdateGrad TBE register"""
    return
