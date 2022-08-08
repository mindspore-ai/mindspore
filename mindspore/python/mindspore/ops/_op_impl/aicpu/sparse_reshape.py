# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""SparseReshape op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sparse_reshape__op_info = AiCPURegOp("SparseReshape") \
    .fusion_type("OPAQUE") \
    .input(0, "indices", "required") \
    .input(1, "shape", "required") \
    .input(2, "new_shape", "required") \
    .output(0, "y_indices", "required") \
    .output(1, "y_shape", "required") \
    .dtype_format(DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default, DataType.I64_Default) \
    .get_op_info()


@op_info_register(sparse_reshape__op_info)
def _sparse_reshape_aicpu():
    """SparseReshape AiCPU register"""
    return
