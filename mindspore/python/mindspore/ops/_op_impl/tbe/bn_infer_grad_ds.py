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

"""BNInferGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bn_infer_grad_op_info = TBERegOp("BNInferGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("bn_infer_grad.so") \
    .compute_cost(10) \
    .kernel_name("bn_infer_grad") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("epsilon", "optional", "float", "all", "0.0001") \
    .input(0, "grads", False, "required", "all", reshape_type="NC") \
    .input(1, "scale", False, "required", "all") \
    .input(2, "batch_variance", False, "required", "all") \
    .output(0, "x_backprop", False, "required", "all", reshape_type="NC") \
    .dtype_format(DataType.F16_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(bn_infer_grad_op_info)
def _bn_infer_grad_ds_tbe():
    """BNInferGrad TBE register"""
    return
