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

"""LRNGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lrn_grad_op_info = TBERegOp("LRNGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("lrn_grad.so") \
    .compute_cost(10) \
    .kernel_name("lrn_grad") \
    .partial_flag(True) \
    .attr("depth_radius", "optional", "int", "all") \
    .attr("bias", "optional", "float", "all") \
    .attr("alpha", "optional", "float", "all") \
    .attr("beta", "optional", "float", "all") \
    .input(0, "grads", False, "required", "all") \
    .input(1, "x", False, "required", "all") \
    .input(2, "y", False, "required", "all") \
    .output(0, "z", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(lrn_grad_op_info)
def _lrn_grad_tbe():
    """LRNGrad TBE register"""
    return
