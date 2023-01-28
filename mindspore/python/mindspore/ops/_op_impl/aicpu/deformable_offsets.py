# Copyright 2023 Huawei Technologies Co., Ltd
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

"""DeformableOffsets op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

deformable_offsets_op_info = AiCPURegOp("DeformableOffsets") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .input(1, "offsets", "required") \
    .output(0, "y", "required") \
    .dtype_format(DataType.F16_NHWC, DataType.F16_NHWC, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC) \
    .get_op_info()


@op_info_register(deformable_offsets_op_info)
def _deformable_offsets_aicpu():
    """DeformableOffsets AiCPU register"""
    return
