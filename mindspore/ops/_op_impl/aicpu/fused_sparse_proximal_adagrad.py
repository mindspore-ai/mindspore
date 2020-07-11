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

"""FusedSparseProximalAdagrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

fused_sparse_proximal_adagrad_op_info = AiCPURegOp("FusedSparseProximalAdagrad") \
    .fusion_type("OPAQUE") \
    .attr("use_locking", "bool") \
    .input(0, "var", "required") \
    .input(1, "accum", "required") \
    .input(2, "lr", "required") \
    .input(3, "l1", "required") \
    .input(4, "l2", "required") \
    .input(5, "grad", "required") \
    .input(6, "indices", "required") \
    .output(0, "var", "required") \
    .output(1, "accum", "required") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.I32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .get_op_info()

@op_info_register(fused_sparse_proximal_adagrad_op_info)
def _fused_sparse_proximal_adagrad_aicpu():
    """FusedSparseProximalAdagrad aicpu register"""
    return
