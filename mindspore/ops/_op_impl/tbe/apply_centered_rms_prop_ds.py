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

"""ApplyCenteredRMSPropD op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

apply_centered_rms_prop_op_info = TBERegOp("ApplyCenteredRMSProp") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("apply_centered_rms_prop_d.so") \
    .compute_cost(10) \
    .kernel_name("apply_centered_rms_prop_d") \
    .dynamic_shape(True) \
    .partial_flag(True) \
    .input(0, "var", False, "required", "all") \
    .input(1, "mg", False, "required", "all") \
    .input(2, "ms", False, "required", "all") \
    .input(3, "mom", False, "required", "all") \
    .input(4, "grad", False, "required", "all") \
    .input(5, "lr", False, "required", "all") \
    .input(6, "rho", False, "required", "all") \
    .input(7, "momentum", False, "required", "all") \
    .input(8, "epsilon", False, "required", "all") \
    .output(0, "var", False, "required", "all") \
    .output(1, "mg", False, "required", "all") \
    .output(2, "ms", False, "required", "all") \
    .output(3, "mom", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD,
                  DataType.F16_5HD, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD,
                  DataType.F16_5HD) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_FracZ,
                  DataType.F16_FracZ, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_FracZ,
                  DataType.F16_FracZ) \
    .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0,
                  DataType.F16_C1HWNCoC0, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0,
                  DataType.F16_C1HWNCoC0) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0,
                  DataType.F32_C1HWNCoC0, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0,
                  DataType.F32_C1HWNCoC0) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .get_op_info()


@op_info_register(apply_centered_rms_prop_op_info)
def _apply_centered_rms_prop_ds_tbe():
    """ApplyCenteredRMSPropD TBE register"""
    return
