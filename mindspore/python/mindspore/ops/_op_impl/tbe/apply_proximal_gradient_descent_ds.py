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

"""ApplyProximalGradientDescent op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

apply_proximal_gradient_descent_ds_op_info = TBERegOp("ApplyProximalGradientDescent") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("apply_proximal_gradient_descent.so") \
    .compute_cost(10) \
    .kernel_name("apply_proximal_gradient_descent") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .input(0, "var", False, "required", "all") \
    .input(1, "alpha", False, "required", "all") \
    .input(2, "l1", False, "required", "all") \
    .input(3, "l2", False, "required", "all") \
    .input(4, "delta", False, "required", "all") \
    .output(0, "var", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_FracZ, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_FracZ, DataType.F32_FracZ) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(apply_proximal_gradient_descent_ds_op_info)
def _apply_proximal_gradient_descent_ds_tbe():
    """ApplyProximalGradientDescent TBE register"""
    return
