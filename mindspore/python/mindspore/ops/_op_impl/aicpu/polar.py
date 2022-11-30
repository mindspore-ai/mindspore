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

"""Polar op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

polar_op_info = AiCPURegOp("Polar") \
    .fusion_type("OPAQUE") \
    .input(0, "abs", "required") \
    .input(1, "angle", "required") \
    .output(0, "y", "required") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.C64_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.C128_Default) \
    .get_op_info()


@op_info_register(polar_op_info)
def _polar_aicpu():
    """Polar aicpu register"""
    return
