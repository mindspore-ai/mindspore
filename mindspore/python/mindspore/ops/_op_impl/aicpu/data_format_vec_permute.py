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

"""DataFormatVecPermute op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
data_format_vec_permute_op_info = AiCPURegOp("DataFormatVecPermute") \
    .fusion_type("OPAQUE") \
    .attr("src_format", "str") \
    .attr("dst_format", "str") \
    .input(0, "x", "required") \
    .output(0, "y", "required") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default) \
    .get_op_info()


@op_info_register(data_format_vec_permute_op_info)
def _data_format_vec_permute_aicpu():
    """DataFormatVecPermute AiCPU register"""
    return
