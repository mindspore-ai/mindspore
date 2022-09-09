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

"""SparseSoftmaxCrossEntropyWithLogitsV2 op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sparse_softmax_cross_entropy_with_logits_v2_op_info = AiCPURegOp("SparseSoftmaxCrossEntropyWithLogitsV2") \
    .fusion_type("OPAQUE") \
    .input(0, "features", "required") \
    .input(1, "labels", "required") \
    .output(0, "loss", "required") \
    .output(1, "backprop", "required") \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(sparse_softmax_cross_entropy_with_logits_v2_op_info)
def _sparse_softmax_cross_entropy_with_logits_v2_aicpu():
    """SparseSoftmaxCrossEntropyWithLogitsV2 AiCPU register"""
    return
