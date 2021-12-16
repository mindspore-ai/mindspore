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
"""HShrinkGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

hshrink_grad_op_info = TBERegOp("HShrinkGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("hard_shrink_grad.so") \
    .compute_cost(10) \
    .kernel_name("hard_shrink_grad") \
    .partial_flag(True) \
    .attr("lambd", "optional", "float", "all", "0.5") \
    .input(0, "gradients", False, "required", "all") \
    .input(1, "features", False, "required", "all") \
    .output(0, "backprops", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(hshrink_grad_op_info)
def _hshrink_grad_tbe():
    """HShrinkGrad TBE register"""
    return
