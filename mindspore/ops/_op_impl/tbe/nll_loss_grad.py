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

"""NLLLossGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

nll_loss_grad_op_info = TBERegOp("NLLLossGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("nll_loss_grad.so") \
    .compute_cost(10) \
    .kernel_name("nll_loss_grad") \
    .partial_flag(True) \
    .attr("reduction", "optional", "str", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "y_grad", False, "required", "all") \
    .input(2, "target", False, "required", "all") \
    .input(3, "weight", False, "required", "all") \
    .input(4, "total_weight", False, "required", "all") \
    .output(0, "x_grad", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(nll_loss_grad_op_info)
def _nll_loss_grad_tbe():
    """NLLLossGrad TBE register"""
    return
