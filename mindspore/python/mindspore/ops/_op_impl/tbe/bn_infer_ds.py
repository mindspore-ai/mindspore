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

"""BNInfer op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bn_infer_op_info = TBERegOp("BNInfer") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("bn_infer.so") \
    .compute_cost(10) \
    .kernel_name("bn_infer") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .dynamic_compile_static(True) \
    .attr("epsilon", "required", "float", "all") \
    .input(0, "x", False, "required", "all", reshape_type="NCH") \
    .input(1, "scale", False, "required", "all") \
    .input(2, "offset", False, "required", "all") \
    .input(3, "mean", False, "required", "all") \
    .input(4, "variance", False, "required", "all") \
    .output(0, "y", False, "required", "all", reshape_type="NCH") \
    .dtype_format(DataType.F16_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(bn_infer_op_info)
def _bn_infer_ds_tbe():
    """BNInfer TBE register"""
    return
