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

"""AvgPoolGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

avg_pool_grad_op_info = TBERegOp("AvgPoolGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("avg_pool_grad_d.so") \
    .compute_cost(10) \
    .kernel_name("avg_pool_grad_d") \
    .partial_flag(True) \
    .attr("x_origin", "required", "listInt", "all", "[]") \
    .attr("kernel_size", "required", "listInt", "all", "[]") \
    .attr("strides", "required", "listInt", "all", "[]") \
    .attr("pad_mode", "required", "str", "all") \
    .attr("format", "optional", "str", "all") \
    .input(0, "input_grad", False, "required", "all") \
    .input(1, "mean_matrix", False, "optional", "all") \
    .input(2, "kernel_matrix", False, "optional", "all") \
    .output(0, "out_grad", True, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_C1HWNCoC0, DataType.F16_5HD) \
    .get_op_info()


@op_info_register(avg_pool_grad_op_info)
def _avg_pool_grad_tbe():
    """AvgPoolGrad TBE register"""
    return
