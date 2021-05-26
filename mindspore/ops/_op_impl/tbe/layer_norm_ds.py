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

"""LayerNorm op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

layer_norm_op_info = TBERegOp("LayerNorm") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("layer_norm.so") \
    .compute_cost(10) \
    .kernel_name("layer_norm") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("begin_norm_axis", "required", "int", "all") \
    .attr("begin_params_axis", "required", "int", "all") \
    .attr("epsilon", "optional", "float", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "gamma", False, "required", "all") \
    .input(2, "beta", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "mean", False, "required", "all") \
    .output(2, "variance", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.F16_None, DataType.F16_None,
                  DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None, DataType.F32_None,
                  DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(layer_norm_op_info)
def _layer_norm_ds_tbe():
    """LayerNorm TBE register"""
    return
