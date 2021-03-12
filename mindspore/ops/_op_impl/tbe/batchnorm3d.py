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

"""BatchNorm3D op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

batch_norm3d_op_info = TBERegOp("BatchNorm3D") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batch_norm3d.so") \
    .compute_cost(10) \
    .kernel_name("batch_norm3d") \
    .partial_flag(True) \
    .attr("epsilon", "optional", "float", "all") \
    .attr("format", "optional", "str", "all") \
    .attr("is_training", "optional", "bool", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "scale", False, "required", "all", reshape_type="C") \
    .input(2, "offset", False, "required", "all", reshape_type="C") \
    .input(3, "mean", False, "optional", "all", reshape_type="C") \
    .input(4, "variance", False, "optional", "all", reshape_type="C") \
    .output(0, "y", False, "required", "all") \
    .output(1, "batch_mean", False, "required", "all") \
    .output(2, "batch_variance", False, "required", "all") \
    .output(3, "reserve_space_1", False, "optional", "all") \
    .output(4, "reserve_space_2", False, "optional", "all") \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0,
                  DataType.F32_NDC1HWC0, DataType.F16_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0,
                  DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0) \
    .dtype_format(DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0,
                  DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0,
                  DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0) \
    .get_op_info()


@op_info_register(batch_norm3d_op_info)
def _batch_norm3d_tbe():
    """BatchNorm3D TBE register"""
    return
