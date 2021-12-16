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

"""LayerNormGrad op"""
from mindspore.ops.op_info_register import op_info_register, CpuRegOp, DataType

layer_norm_grad_op_info = CpuRegOp("LayerNormGrad") \
    .input(0, "dy", "required") \
    .input(1, "x", "required") \
    .input(2, "variance", "required") \
    .input(3, "mean", "required") \
    .input(4, "gamma", "required") \
    .output(0, "pd_x", "required") \
    .output(1, "pd_gamma", "required") \
    .output(2, "pd_beta", "required") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(layer_norm_grad_op_info)
def _layer_norm_grad_cpu():
    """LayerNormGrad TBE register"""
    return
