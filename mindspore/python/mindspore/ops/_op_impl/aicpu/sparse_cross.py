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

"""SparseCross op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sparse_cross_op_info = AiCPURegOp("SparseCross")                                      \
    .fusion_type("OPAQUE")                                                            \
    .attr("N", "int")                                                                  \
    .attr("hashed_output", "bool")                                                    \
    .attr("hash_key", "int")                                                          \
    .attr("out_type", "Type")                                                         \
    .attr("internal_type", "Type")                                                    \
    .attr("num_buckets", "int")                                                       \
    .input(0, "indices", "dynamic")                                                   \
    .input(1, "values", "dynamic")                                                    \
    .input(2, "shapes", "dynamic")                                                    \
    .input(3, "dense_inputs", "dynamic")                                              \
    .output(0, "output_indices", "required")                                          \
    .output(1, "output_values", "required")                                           \
    .output(2, "output_shape", "required")                                            \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default) \
    .get_op_info()


@op_info_register(sparse_cross_op_info)
def _sparse_cross_aicpu():
    """SparseCross AiCPU register"""
    return
